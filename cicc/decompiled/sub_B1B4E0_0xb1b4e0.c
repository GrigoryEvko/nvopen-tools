// Function: sub_B1B4E0
// Address: 0xb1b4e0
//
__int64 __fastcall sub_B1B4E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 *v12; // r15
  __int64 v13; // r8
  __int64 v14; // rdi
  int v15; // r15d
  __int64 v16; // [rsp+8h] [rbp-48h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v4 = a1 + 16;
  result = sub_C8D7D0(a1, a1 + 16, a2, 8, v17);
  v7 = *(__int64 **)a1;
  v8 = result;
  v9 = 8LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = result + v9;
    do
    {
      if ( result )
      {
        v4 = *v7;
        *(_QWORD *)result = *v7;
        *v7 = 0;
      }
      result += 8;
      ++v7;
    }
    while ( result != v11 );
    v12 = *(__int64 **)a1;
    result = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)a1 + 8 * result;
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v13 = *(_QWORD *)(v10 - 8);
        v10 -= 8;
        if ( v13 )
        {
          v14 = *(_QWORD *)(v13 + 24);
          if ( v14 != v13 + 40 )
          {
            v16 = v13;
            _libc_free(v14, v4);
            v13 = v16;
          }
          v4 = 80;
          result = j_j___libc_free_0(v13, 80);
        }
      }
      while ( v12 != (__int64 *)v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v15 = v17[0];
  if ( v3 != v10 )
    result = _libc_free(v10, v4);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v15;
  return result;
}
