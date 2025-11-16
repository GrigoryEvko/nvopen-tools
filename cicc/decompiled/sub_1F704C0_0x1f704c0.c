// Function: sub_1F704C0
// Address: 0x1f704c0
//
__int64 __fastcall sub_1F704C0(__int64 **a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // r13
  const void **v11; // r15
  __int64 v12; // rcx
  int v13; // eax
  char v14; // al
  __int64 v15; // rsi
  __int64 *v16; // r10
  __int64 v17; // r13
  __int128 v19; // [rsp-10h] [rbp-60h]
  __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 *v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h] BYREF
  int v24; // [rsp+18h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = *v7;
  v9 = *v7;
  v10 = v7[1];
  v11 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v12 = **(unsigned __int8 **)(a2 + 40);
  v13 = *(unsigned __int16 *)(*v7 + 24);
  if ( v13 != 11 && v13 != 33 )
  {
    v21 = **(unsigned __int8 **)(a2 + 40);
    v14 = sub_1D16930(v8);
    v12 = v21;
    if ( !v14 )
      return 0;
  }
  v15 = *(_QWORD *)(a2 + 72);
  v16 = *a1;
  v23 = v15;
  if ( v15 )
  {
    v20 = v12;
    v22 = v16;
    sub_1623A60((__int64)&v23, v15, 2);
    v12 = v20;
    v16 = v22;
  }
  *((_QWORD *)&v19 + 1) = v10;
  *(_QWORD *)&v19 = v9;
  v24 = *(_DWORD *)(a2 + 64);
  v17 = sub_1D309E0(v16, 179, (__int64)&v23, v12, v11, 0, a3, a4, a5, v19);
  if ( v23 )
    sub_161E7C0((__int64)&v23, v23);
  return v17;
}
