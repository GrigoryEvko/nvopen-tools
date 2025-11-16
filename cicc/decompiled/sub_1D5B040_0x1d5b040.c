// Function: sub_1D5B040
// Address: 0x1d5b040
//
__int64 __fastcall sub_1D5B040(__int64 a1, char a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rdx

  v3 = 0;
  while ( a1 )
  {
    v6 = *(_QWORD **)(a3 + 16);
    v7 = *(_QWORD **)(a3 + 8);
    if ( v6 == v7 )
    {
      v8 = &v7[*(unsigned int *)(a3 + 28)];
      if ( v7 == v8 )
      {
        v13 = *(_QWORD **)(a3 + 8);
      }
      else
      {
        do
        {
          if ( a1 == *v7 )
            break;
          ++v7;
        }
        while ( v8 != v7 );
        v13 = v8;
      }
    }
    else
    {
      v8 = &v6[*(unsigned int *)(a3 + 24)];
      v7 = sub_16CC9F0(a3, a1);
      if ( a1 == *v7 )
      {
        v11 = *(_QWORD *)(a3 + 16);
        if ( v11 == *(_QWORD *)(a3 + 8) )
          v12 = *(unsigned int *)(a3 + 28);
        else
          v12 = *(unsigned int *)(a3 + 24);
        v13 = (_QWORD *)(v11 + 8 * v12);
      }
      else
      {
        v9 = *(_QWORD *)(a3 + 16);
        if ( v9 != *(_QWORD *)(a3 + 8) )
        {
          v7 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(a3 + 24));
          goto LABEL_7;
        }
        v7 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(a3 + 28));
        v13 = v7;
      }
    }
    while ( v13 != v7 && *v7 >= 0xFFFFFFFFFFFFFFFELL )
      ++v7;
LABEL_7:
    if ( v7 == v8 )
      return v3;
    if ( a2 )
    {
      v3 = *(_QWORD *)(a1 - 48);
      if ( *(_BYTE *)(v3 + 16) != 79 )
        return v3;
    }
    else
    {
      v3 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v3 + 16) != 79 )
        return v3;
    }
    a1 = v3;
  }
  return v3;
}
