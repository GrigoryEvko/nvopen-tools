// Function: sub_AE4030
// Address: 0xae4030
//
__int64 __fastcall sub_AE4030(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rdi
  __int64 v5; // rsi
  _QWORD *v6; // rdi
  _QWORD *v7; // rdi
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  _QWORD *v10; // rdi
  __int64 result; // rax
  _QWORD *v12; // rdi
  _QWORD *v13; // r12
  _QWORD *v14; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rdi

  v3 = a1[61];
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    v5 = 16LL * *(unsigned int *)(v3 + 24);
    if ( *(_DWORD *)(v3 + 16) )
    {
      v13 = (_QWORD *)(v4 + v5);
      if ( v4 + v5 != v4 )
      {
        v14 = *(_QWORD **)(v3 + 8);
        while ( 1 )
        {
          v15 = v14;
          if ( *v14 != -4096 && *v14 != -8192 )
            break;
          v14 += 2;
          if ( v13 == v14 )
            goto LABEL_3;
        }
        if ( v14 != v13 )
        {
          do
          {
            v16 = v15[1];
            v15 += 2;
            _libc_free(v16, v5);
            if ( v15 == v13 )
              break;
            while ( *v15 == -8192 || *v15 == -4096 )
            {
              v15 += 2;
              if ( v13 == v15 )
                goto LABEL_28;
            }
          }
          while ( v13 != v15 );
LABEL_28:
          v4 = *(_QWORD *)(v3 + 8);
          v5 = 16LL * *(unsigned int *)(v3 + 24);
        }
      }
    }
LABEL_3:
    sub_C7D6A0(v4, v5, 8);
    a2 = 32;
    j_j___libc_free_0(v3, 32);
  }
  v6 = (_QWORD *)a1[56];
  if ( v6 != a1 + 58 )
  {
    a2 = a1[58] + 1LL;
    j_j___libc_free_0(v6, a2);
  }
  v7 = (_QWORD *)a1[34];
  if ( v7 != a1 + 36 )
    _libc_free(v7, a2);
  v8 = (_QWORD *)a1[22];
  if ( v8 != a1 + 24 )
    _libc_free(v8, a2);
  v9 = (_QWORD *)a1[16];
  if ( v9 != a1 + 18 )
    _libc_free(v9, a2);
  v10 = (_QWORD *)a1[8];
  result = (__int64)(a1 + 10);
  if ( v10 != a1 + 10 )
    result = _libc_free(v10, a2);
  v12 = (_QWORD *)a1[4];
  if ( v12 != a1 + 7 )
    return _libc_free(v12, a2);
  return result;
}
