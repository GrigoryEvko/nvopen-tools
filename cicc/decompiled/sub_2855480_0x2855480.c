// Function: sub_2855480
// Address: 0x2855480
//
__int64 __fastcall sub_2855480(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  unsigned int v11; // r8d
  __int64 *v13; // rax
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *i; // rdx
  __int64 v17; // [rsp+8h] [rbp-38h]

  v3 = (_BYTE *)*a1;
  if ( *(_BYTE *)*a1 == 84 )
  {
    v4 = 0;
    v5 = a2 + 56;
    v6 = 8LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
    if ( (*((_DWORD *)v3 + 1) & 0x7FFFFFF) == 0 )
      return 1;
    while ( 1 )
    {
      v7 = *((_QWORD *)v3 - 1);
      if ( a1[1] == *(_QWORD *)(v7 + 4 * v4) )
      {
        v8 = *(_QWORD *)(32LL * *((unsigned int *)v3 + 18) + v7 + v4);
        if ( *(_BYTE *)(a2 + 84) )
        {
          v9 = *(_QWORD **)(a2 + 64);
          v10 = &v9[*(unsigned int *)(a2 + 76)];
          if ( v9 != v10 )
          {
            while ( v8 != *v9 )
            {
              if ( v10 == ++v9 )
                goto LABEL_4;
            }
            return 0;
          }
        }
        else
        {
          v17 = v5;
          v13 = sub_C8CA60(v5, v8);
          v5 = v17;
          if ( v13 )
            return 0;
        }
      }
LABEL_4:
      v4 += 8;
      if ( v4 == v6 )
        return 1;
    }
  }
  v11 = *(unsigned __int8 *)(a2 + 84);
  v14 = *((_QWORD *)v3 + 5);
  if ( (_BYTE)v11 )
  {
    v15 = *(_QWORD **)(a2 + 64);
    for ( i = &v15[*(unsigned int *)(a2 + 76)]; i != v15; ++v15 )
    {
      if ( v14 == *v15 )
        return 0;
    }
  }
  else
  {
    LOBYTE(v11) = sub_C8CA60(a2 + 56, v14) == 0;
  }
  return v11;
}
