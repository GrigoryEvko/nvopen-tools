// Function: sub_29D0220
// Address: 0x29d0220
//
void __fastcall sub_29D0220(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 *v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rcx
  __int64 *v10; // r12
  _QWORD *v11; // rcx
  __int64 *v12; // r15
  int v13; // r15d
  unsigned __int64 v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v14, a6);
  v7 = *(__int64 **)a1;
  v8 = v6;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = (__int64 *)(*(_QWORD *)a1 + v9 * 8);
  if ( *(__int64 **)a1 != v10 )
  {
    v11 = &v6[v9];
    do
    {
      if ( v6 )
      {
        *v6 = 0;
        *v6 = *v7;
        *v7 = 0;
      }
      ++v6;
      ++v7;
    }
    while ( v6 != v11 );
    v12 = *(__int64 **)a1;
    v10 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    if ( *(__int64 **)a1 != v10 )
    {
      do
        sub_29CF750(--v10);
      while ( v12 != v10 );
      v10 = *(__int64 **)a1;
    }
  }
  v13 = v14[0];
  if ( (__int64 *)(a1 + 16) != v10 )
    _libc_free((unsigned __int64)v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v13;
}
