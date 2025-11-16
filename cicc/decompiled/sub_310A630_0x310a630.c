// Function: sub_310A630
// Address: 0x310a630
//
void __fastcall sub_310A630(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rsi
  __int64 v5; // rsi
  unsigned int v6; // r8d
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 *v9; // rdi
  _QWORD *v10; // rax
  bool v11; // cc
  unsigned __int64 v12; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-68h]
  unsigned __int64 v14; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-58h]
  unsigned __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-48h]
  unsigned __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_WORD *)(v2 + 24) )
    return;
  v3 = *(_QWORD *)(a2 + 32);
  v13 = *(_DWORD *)(v3 + 32);
  if ( v13 > 0x40 )
    sub_C43780((__int64)&v12, (const void **)(v3 + 24));
  else
    v12 = *(_QWORD *)(v3 + 24);
  v5 = *(_QWORD *)(v2 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  v15 = v6;
  if ( v6 > 0x40 )
  {
    sub_C43780((__int64)&v14, (const void **)(v5 + 24));
    v6 = v15;
  }
  else
  {
    v14 = *(_QWORD *)(v5 + 24);
  }
  v7 = v13;
  if ( v13 > v6 )
  {
    sub_C44830((__int64)&v18, &v14, v13);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    v7 = v13;
    v14 = v18;
    v15 = v19;
  }
  else if ( v13 < v6 )
  {
    sub_C44830((__int64)&v18, &v12, v6);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    v7 = v19;
    v12 = v18;
    v13 = v19;
  }
  v17 = v7;
  if ( v7 <= 0x40 )
  {
    v16 = 0;
    v19 = v7;
LABEL_10:
    v18 = 0;
    goto LABEL_11;
  }
  sub_C43690((__int64)&v16, 0, 0);
  v19 = v13;
  if ( v13 <= 0x40 )
    goto LABEL_10;
  sub_C43690((__int64)&v18, 0, 0);
LABEL_11:
  sub_C4C400((__int64)&v12, (__int64)&v14, (__int64)&v16, (__int64)&v18);
  v8 = sub_DA26C0(*(__int64 **)a1, (__int64)&v16);
  v9 = *(__int64 **)a1;
  *(_QWORD *)(a1 + 16) = v8;
  v10 = sub_DA26C0(v9, (__int64)&v18);
  v11 = v19 <= 0x40;
  *(_QWORD *)(a1 + 24) = v10;
  if ( !v11 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 )
  {
    if ( v12 )
      j_j___libc_free_0_0(v12);
  }
}
