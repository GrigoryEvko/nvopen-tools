// Function: sub_37C5040
// Address: 0x37c5040
//
void __fastcall sub_37C5040(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int64 *v13; // r12
  __int64 v14; // r15
  __int64 v15; // rdx
  unsigned __int64 *v16; // r15
  int v17; // r15d
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x250u, v19, a6);
  v9 = *(unsigned int *)(a1 + 8);
  v10 = v6;
  v11 = *(_QWORD *)a1;
  v12 = 9 * v9;
  v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 592 * v9);
  if ( *(unsigned __int64 **)a1 != v13 )
  {
    v14 = v10;
    do
    {
      if ( v14 )
      {
        *(_DWORD *)(v14 + 8) = 0;
        *(_QWORD *)v14 = v14 + 16;
        *(_DWORD *)(v14 + 12) = 8;
        v15 = *(unsigned int *)(v11 + 8);
        if ( (_DWORD)v15 )
        {
          v18 = v11;
          sub_37B73F0(v14, v11, v15, v12, v7, v8);
          v11 = v18;
        }
      }
      v11 += 592;
      v14 += 592;
    }
    while ( v13 != (unsigned __int64 *)v11 );
    v16 = *(unsigned __int64 **)a1;
    v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 592LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v13 )
    {
      do
      {
        v13 -= 74;
        if ( (unsigned __int64 *)*v13 != v13 + 2 )
          _libc_free(*v13);
      }
      while ( v13 != v16 );
      v13 = *(unsigned __int64 **)a1;
    }
  }
  v17 = v19[0];
  if ( (unsigned __int64 *)(a1 + 16) != v13 )
    _libc_free((unsigned __int64)v13);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v17;
}
