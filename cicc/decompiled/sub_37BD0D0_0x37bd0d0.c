// Function: sub_37BD0D0
// Address: 0x37bd0d0
//
void __fastcall sub_37BD0D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx
  _QWORD *v12; // r15
  unsigned __int64 *v13; // r8
  int v14; // r15d
  unsigned __int64 *v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v16, a6);
  v7 = *(_QWORD **)a1;
  v8 = v6;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9 * 8;
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = &v6[v9];
    do
    {
      if ( v6 )
      {
        *v6 = *v7;
        *v7 = 0;
      }
      ++v6;
      ++v7;
    }
    while ( v6 != v11 );
    v12 = *(_QWORD **)a1;
    v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v13 = *(unsigned __int64 **)(v10 - 8);
        v10 -= 8LL;
        if ( v13 )
        {
          if ( (unsigned __int64 *)*v13 != v13 + 2 )
          {
            v15 = v13;
            _libc_free(*v13);
            v13 = v15;
          }
          j_j___libc_free_0((unsigned __int64)v13);
        }
      }
      while ( v12 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v14 = v16[0];
  if ( a1 + 16 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v14;
}
