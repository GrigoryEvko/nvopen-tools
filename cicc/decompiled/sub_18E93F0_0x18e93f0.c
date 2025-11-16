// Function: sub_18E93F0
// Address: 0x18e93f0
//
void __fastcall sub_18E93F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v7; // r12
  __int64 v8; // r15
  __int64 *v9; // r14
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r9d
  __int64 *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rdi
  __int64 *v21; // rbx
  __int64 *v22; // rsi
  __int64 v23; // rsi
  _QWORD *v24; // rax
  __int64 *v25; // r8
  char v26; // [rsp+0h] [rbp-50h]
  __int64 *v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+18h] [rbp-38h]

  v7 = *(__int64 **)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 40);
  v9 = v7;
  if ( (__int64 *)v8 == v7 )
    goto LABEL_6;
  _BitScanReverse64(&v10, 0xCCCCCCCCCCCCCCCDLL * ((v8 - (__int64)v7) >> 5));
  sub_18E7670(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 40), 2LL * (int)(63 - (v10 ^ 0x3F)), a4, a5, a6, v26);
  if ( v8 - (__int64)v7 <= 2560 )
  {
    sub_18E6910((__int64)v7, v8, v11, v12, v13, v14);
    goto LABEL_21;
  }
  v15 = v7 + 320;
  sub_18E6910((__int64)v7, (__int64)(v7 + 320), v11, v12, v13, v14);
  if ( (__int64 *)v8 == v7 + 320 )
  {
LABEL_21:
    v7 = *(__int64 **)(a1 + 32);
    v9 = *(__int64 **)(a1 + 40);
    goto LABEL_6;
  }
  do
  {
    v20 = (__int64)v15;
    v15 += 20;
    sub_18E67F0(v20, (__int64)(v7 + 320), v16, v17, v18, v19);
  }
  while ( (__int64 *)v8 != v15 );
  v7 = *(__int64 **)(a1 + 32);
  v9 = *(__int64 **)(a1 + 40);
LABEL_6:
  v21 = v7 + 20;
  if ( v7 + 20 != v9 )
  {
    while ( 1 )
    {
      v23 = v21[18];
      v24 = (_QWORD *)v7[18];
      if ( *v24 != *(_QWORD *)v23 )
        goto LABEL_8;
      v25 = v24 + 3;
      v29 = *(_DWORD *)(v23 + 32);
      if ( v29 > 0x40 )
      {
        v27 = v24 + 3;
        sub_16A4FD0((__int64)&v28, (const void **)(v23 + 24));
        v25 = v27;
      }
      else
      {
        v28 = *(_QWORD *)(v23 + 24);
      }
      sub_16A7590((__int64)&v28, v25);
      if ( v29 > 0x40 )
      {
        if ( v28 )
          j_j___libc_free_0_0(v28);
        goto LABEL_8;
      }
      if ( (unsigned __int8)sub_14A2A30(*(_QWORD *)a1) )
      {
        v21 += 20;
        if ( v9 == v21 )
        {
LABEL_15:
          v9 = *(__int64 **)(a1 + 40);
          break;
        }
      }
      else
      {
LABEL_8:
        v22 = v7;
        v7 = v21;
        sub_18E8FE0(a1, v22, v21);
        v21 += 20;
        if ( v9 == v21 )
          goto LABEL_15;
      }
    }
  }
  sub_18E8FE0(a1, v7, v9);
}
