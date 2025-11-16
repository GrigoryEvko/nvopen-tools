// Function: sub_1F7F610
// Address: 0x1f7f610
//
__int64 __fastcall sub_1F7F610(__int64 **a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r10
  __int64 v10; // r11
  const void **v11; // r13
  __int64 v12; // rcx
  int v13; // eax
  char v14; // al
  __int64 v15; // rsi
  __int64 *v16; // r15
  __int64 v17; // r12
  __int128 v19; // [rsp-10h] [rbp-70h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = *v7;
  v9 = *v7;
  v10 = v7[1];
  v11 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v12 = **(unsigned __int8 **)(a2 + 40);
  v13 = *(unsigned __int16 *)(*v7 + 24);
  if ( v13 != 11 && v13 != 33 )
  {
    v20 = **(unsigned __int8 **)(a2 + 40);
    v22 = v9;
    v24 = v10;
    v14 = sub_1D16930(v8);
    v9 = v22;
    v10 = v24;
    v12 = v20;
    if ( !v14 )
      return sub_1F7EF90(a2, *a1, a3, a4, a5);
  }
  v15 = *(_QWORD *)(a2 + 72);
  v16 = *a1;
  v26 = v15;
  if ( v15 )
  {
    v21 = v12;
    v23 = v9;
    v25 = v10;
    sub_1623A60((__int64)&v26, v15, 2);
    v12 = v21;
    v9 = v23;
    v10 = v25;
  }
  *((_QWORD *)&v19 + 1) = v10;
  *(_QWORD *)&v19 = v9;
  v27 = *(_DWORD *)(a2 + 64);
  v17 = sub_1D309E0(v16, 153, (__int64)&v26, v12, v11, 0, a3, a4, a5, v19);
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v17;
}
