// Function: sub_1F71260
// Address: 0x1f71260
//
__int64 __fastcall sub_1F71260(__int64 **a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r15
  const void **v10; // r8
  __int64 v11; // rcx
  int v12; // eax
  char v13; // al
  __int64 v14; // rsi
  __int64 *v15; // r13
  __int64 v16; // rax
  __int64 v17; // r14
  int v19; // eax
  __int64 v20; // rsi
  __int64 *v21; // r13
  __int128 *v22; // r14
  __int128 v23; // [rsp-10h] [rbp-70h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  const void **v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  const void **v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h] BYREF
  int v30; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *v6;
  v8 = *v6;
  v9 = v6[1];
  v10 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v11 = **(unsigned __int8 **)(a2 + 40);
  v12 = *(unsigned __int16 *)(*v6 + 24);
  if ( v12 == 11
    || v12 == 33
    || (v24 = **(unsigned __int8 **)(a2 + 40),
        v25 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
        v27 = v7,
        v13 = sub_1D16930(v7),
        v10 = v25,
        v11 = v24,
        v13) )
  {
    v14 = *(_QWORD *)(a2 + 72);
    v15 = *a1;
    v29 = v14;
    if ( v14 )
    {
      v26 = v11;
      v28 = v10;
      sub_1623A60((__int64)&v29, v14, 2);
      v11 = v26;
      v10 = v28;
    }
    *((_QWORD *)&v23 + 1) = v9;
    *(_QWORD *)&v23 = v8;
    v30 = *(_DWORD *)(a2 + 64);
    v16 = sub_1D309E0(v15, 163, (__int64)&v29, v11, v10, 0, a3, a4, a5, v23);
LABEL_7:
    v17 = v16;
    if ( v29 )
      sub_161E7C0((__int64)&v29, v29);
    return v17;
  }
  v19 = *(unsigned __int16 *)(v27 + 24);
  if ( v19 == 163 )
    return **(_QWORD **)(a2 + 32);
  if ( v19 == 162 || (v17 = 0, v19 == 101) )
  {
    v20 = *(_QWORD *)(a2 + 72);
    v21 = *a1;
    v22 = *(__int128 **)(v27 + 32);
    v29 = v20;
    if ( v20 )
    {
      sub_1623A60((__int64)&v29, v20, 2);
      v11 = v24;
      v10 = v25;
    }
    v30 = *(_DWORD *)(a2 + 64);
    v16 = sub_1D309E0(v21, 163, (__int64)&v29, v11, v10, 0, a3, a4, a5, *v22);
    goto LABEL_7;
  }
  return v17;
}
