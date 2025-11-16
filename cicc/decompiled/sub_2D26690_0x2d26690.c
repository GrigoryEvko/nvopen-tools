// Function: sub_2D26690
// Address: 0x2d26690
//
__int64 __fastcall sub_2D26690(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r13
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // r15
  __int64 v20; // rsi
  int v21; // r13d
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  unsigned __int64 v25; // [rsp+18h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v24 = a1 + 16;
  v23 = sub_C8D7D0(a1, a1 + 16, a2, 0x48u, v26, a6);
  v10 = v23;
  v11 = *(_QWORD *)a1;
  v12 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    do
    {
      while ( 1 )
      {
        if ( v10 )
        {
          v13 = *(_QWORD *)v11;
          *(_DWORD *)(v10 + 16) = 0;
          *(_DWORD *)(v10 + 20) = 2;
          *(_QWORD *)v10 = v13;
          *(_QWORD *)(v10 + 8) = v10 + 24;
          v14 = *(unsigned int *)(v11 + 16);
          if ( (_DWORD)v14 )
            break;
        }
        v11 += 72LL;
        v10 += 72;
        if ( v12 == v11 )
          goto LABEL_7;
      }
      v15 = v10 + 8;
      v25 = v11;
      v10 += 72;
      sub_2D262B0(v15, v11 + 8, v14, v7, v8, v9);
      v11 = v25 + 72;
    }
    while ( v12 != v25 + 72 );
LABEL_7:
    v16 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v17 = *(unsigned int *)(v12 - 56);
        v18 = *(_QWORD *)(v12 - 64);
        v12 -= 72LL;
        v19 = v18 + 24 * v17;
        if ( v18 != v19 )
        {
          do
          {
            v20 = *(_QWORD *)(v19 - 8);
            v19 -= 24LL;
            if ( v20 )
              sub_B91220(v19 + 16, v20);
          }
          while ( v18 != v19 );
          v18 = *(_QWORD *)(v12 + 8);
        }
        if ( v18 != v12 + 24 )
          _libc_free(v18);
      }
      while ( v12 != v16 );
      v12 = *(_QWORD *)a1;
    }
  }
  v21 = v26[0];
  if ( v24 != v12 )
    _libc_free(v12);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return v23;
}
