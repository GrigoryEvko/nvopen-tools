// Function: sub_25126F0
// Address: 0x25126f0
//
void __fastcall sub_25126F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  __int64 v10; // rcx
  unsigned __int64 v11; // r14
  _QWORD *v12; // rcx
  _QWORD *v13; // r15
  _QWORD *v14; // r12
  void (__fastcall *v15)(_QWORD *, _QWORD *, __int64); // rax
  void (__fastcall *v16)(_QWORD *, _QWORD *, __int64); // rax
  unsigned __int64 v17; // rdi
  int v18; // r12d
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v19 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v20, a6);
  v8 = *(_QWORD **)a1;
  v9 = v7;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1 + v10 * 8;
  if ( *(_QWORD *)a1 != v11 )
  {
    v12 = &v7[v10];
    do
    {
      if ( v7 )
      {
        *v7 = *v8;
        *v8 = 0;
      }
      ++v7;
      ++v8;
    }
    while ( v7 != v12 );
    v13 = *(_QWORD **)a1;
    v11 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        v14 = *(_QWORD **)(v11 - 8);
        v11 -= 8LL;
        if ( v14 )
        {
          v15 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v14[19];
          if ( v15 )
            v15(v14 + 17, v14 + 17, 3);
          v16 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v14[15];
          if ( v16 )
            v16(v14 + 13, v14 + 13, 3);
          v17 = v14[3];
          if ( (_QWORD *)v17 != v14 + 5 )
            _libc_free(v17);
          j_j___libc_free_0((unsigned __int64)v14);
        }
      }
      while ( v13 != (_QWORD *)v11 );
      v11 = *(_QWORD *)a1;
    }
  }
  v18 = v20[0];
  if ( v19 != v11 )
    _libc_free(v11);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v18;
}
