// Function: sub_31B5960
// Address: 0x31b5960
//
void __fastcall sub_31B5960(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx
  _QWORD *v12; // r14
  unsigned __int64 v13; // r15
  int v14; // r15d
  unsigned __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v15, a6);
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
        v13 = *(_QWORD *)(v10 - 8);
        v10 -= 8LL;
        if ( v13 )
        {
          sub_371BB90(v13);
          j_j___libc_free_0(v13);
        }
      }
      while ( v12 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v14 = v15[0];
  if ( a1 + 16 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v14;
}
