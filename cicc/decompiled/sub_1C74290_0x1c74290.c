// Function: sub_1C74290
// Address: 0x1c74290
//
__int64 __fastcall sub_1C74290(_QWORD **a1, __int64 a2)
{
  _QWORD *v2; // rax
  unsigned int v3; // r8d
  _QWORD *v4; // rdx
  unsigned int v5; // r8d
  __int64 v6; // r13
  __int64 v7; // rbx
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD *v15; // rdx

  v2 = sub_1648700(a2);
  v4 = *a1;
  if ( *((_BYTE *)v2 + 16) > 0x17u )
  {
    v5 = 0;
    if ( v2 == (_QWORD *)*v4 )
      return v5;
    v6 = v2[5];
    v7 = *a1[1];
    v8 = *(_QWORD **)(v7 + 72);
    v9 = *(_QWORD **)(v7 + 64);
    if ( v8 == v9 )
    {
      v10 = &v9[*(unsigned int *)(v7 + 84)];
      if ( v9 == v10 )
      {
        v15 = *(_QWORD **)(v7 + 64);
      }
      else
      {
        do
        {
          if ( v6 == *v9 )
            break;
          ++v9;
        }
        while ( v10 != v9 );
        v15 = v10;
      }
    }
    else
    {
      v10 = &v8[*(unsigned int *)(v7 + 80)];
      v9 = sub_16CC9F0(v7 + 56, v6);
      if ( v6 == *v9 )
      {
        v13 = *(_QWORD *)(v7 + 72);
        if ( v13 == *(_QWORD *)(v7 + 64) )
          v14 = *(unsigned int *)(v7 + 84);
        else
          v14 = *(unsigned int *)(v7 + 80);
        v15 = (_QWORD *)(v13 + 8 * v14);
      }
      else
      {
        v11 = *(_QWORD *)(v7 + 72);
        if ( v11 != *(_QWORD *)(v7 + 64) )
        {
          v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(v7 + 80));
LABEL_7:
          LOBYTE(v5) = v10 == v9;
          return v5;
        }
        v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(v7 + 84));
        v15 = v9;
      }
    }
    while ( v15 != v9 && *v9 >= 0xFFFFFFFFFFFFFFFELL )
      ++v9;
    goto LABEL_7;
  }
  LOBYTE(v3) = *v4 != (_QWORD)v2;
  return v3;
}
