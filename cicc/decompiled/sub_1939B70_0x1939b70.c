// Function: sub_1939B70
// Address: 0x1939b70
//
char __fastcall sub_1939B70(_QWORD **a1, __int64 *a2)
{
  char result; // al
  __int64 v3; // rbx
  __int64 v4; // r12
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rdx

  result = 1;
  v3 = *a2;
  if ( *a2 != **a1 )
  {
    v4 = *a1[1];
    v5 = *(_QWORD **)(v4 + 72);
    v6 = *(_QWORD **)(v4 + 64);
    if ( v5 == v6 )
    {
      v7 = &v6[*(unsigned int *)(v4 + 84)];
      if ( v6 == v7 )
      {
        v11 = *(_QWORD **)(v4 + 64);
      }
      else
      {
        do
        {
          if ( v3 == *v6 )
            break;
          ++v6;
        }
        while ( v7 != v6 );
        v11 = v7;
      }
LABEL_15:
      while ( v11 != v6 )
      {
        if ( *v6 < 0xFFFFFFFFFFFFFFFELL )
          return v7 != v6;
        ++v6;
      }
      return v7 != v6;
    }
    else
    {
      v7 = &v5[*(unsigned int *)(v4 + 80)];
      v6 = sub_16CC9F0(v4 + 56, *a2);
      if ( v3 == *v6 )
      {
        v9 = *(_QWORD *)(v4 + 72);
        if ( v9 == *(_QWORD *)(v4 + 64) )
          v10 = *(unsigned int *)(v4 + 84);
        else
          v10 = *(unsigned int *)(v4 + 80);
        v11 = (_QWORD *)(v9 + 8 * v10);
        goto LABEL_15;
      }
      v8 = *(_QWORD *)(v4 + 72);
      if ( v8 == *(_QWORD *)(v4 + 64) )
      {
        v6 = (_QWORD *)(v8 + 8LL * *(unsigned int *)(v4 + 84));
        v11 = v6;
        goto LABEL_15;
      }
      v6 = (_QWORD *)(v8 + 8LL * *(unsigned int *)(v4 + 80));
      return v7 != v6;
    }
  }
  return result;
}
