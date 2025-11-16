// Function: sub_2DD0E60
// Address: 0x2dd0e60
//
void __fastcall sub_2DD0E60(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  _QWORD *v9; // rax
  __int64 *v10; // rdx
  _QWORD *v11; // r13
  __int64 v12; // rcx
  unsigned __int64 v13; // r14
  _QWORD *v14; // rcx
  __int64 *v15; // r15
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r12
  __int64 (__fastcall *v18)(_QWORD *); // rax
  int v19; // r12d
  __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v20 = a1 + 16;
  v9 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v21, a6);
  v10 = *(__int64 **)a1;
  v11 = v9;
  v12 = *(unsigned int *)(a1 + 8);
  v13 = *(_QWORD *)a1 + v12 * 8;
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = &v9[v12];
    do
    {
      if ( v9 )
      {
        v7 = *v10;
        *v9 = *v10;
        *v10 = 0;
      }
      ++v9;
      ++v10;
    }
    while ( v9 != v14 );
    v15 = *(__int64 **)a1;
    v13 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = *(_QWORD **)(v13 - 8);
          v13 -= 8LL;
          if ( v17 )
            break;
LABEL_11:
          if ( v15 == (__int64 *)v13 )
            goto LABEL_15;
        }
        v18 = *(__int64 (__fastcall **)(_QWORD *))(*v17 + 8LL);
        if ( v18 == sub_BD9990 )
        {
          v16 = v17[1];
          *v17 = &unk_49DB390;
          if ( (_QWORD *)v16 != v17 + 3 )
            j_j___libc_free_0(v16);
          v7 = 48;
          j_j___libc_free_0((unsigned __int64)v17);
          goto LABEL_11;
        }
        ((void (__fastcall *)(_QWORD *, __int64, __int64 *))v18)(v17, v7, v10);
        if ( v15 == (__int64 *)v13 )
        {
LABEL_15:
          v13 = *(_QWORD *)a1;
          break;
        }
      }
    }
  }
  v19 = v21[0];
  if ( v20 != v13 )
    _libc_free(v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v19;
}
