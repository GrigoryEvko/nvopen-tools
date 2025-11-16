// Function: sub_AE4800
// Address: 0xae4800
//
__int64 __fastcall sub_AE4800(unsigned int *a1, __int64 a2)
{
  unsigned int *v3; // rsi
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // r12
  _DWORD *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r15
  unsigned int v11; // r15d
  __int64 v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 4;
  result = sub_C8D7D0(a1, a1 + 4, a2, 16, v12);
  v5 = result;
  v6 = 16LL * a1[2];
  v7 = *(_QWORD *)a1 + v6;
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = (_DWORD *)(*(_QWORD *)a1 + 8LL);
    v9 = result + v6;
    do
    {
      if ( result )
      {
        *(_DWORD *)(result + 8) = *v8;
        v3 = (unsigned int *)*((_QWORD *)v8 - 1);
        *(_QWORD *)result = v3;
        *v8 = 0;
      }
      result += 16;
      v8 += 4;
    }
    while ( result != v9 );
    v10 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 16LL * a1[2];
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v7 -= 16;
        if ( *(_DWORD *)(v7 + 8) > 0x40u && *(_QWORD *)v7 )
          result = j_j___libc_free_0_0(*(_QWORD *)v7);
      }
      while ( v10 != v7 );
      v7 = *(_QWORD *)a1;
    }
  }
  v11 = v12[0];
  if ( a1 + 4 != (unsigned int *)v7 )
    result = _libc_free(v7, v3);
  *(_QWORD *)a1 = v5;
  a1[3] = v11;
  return result;
}
