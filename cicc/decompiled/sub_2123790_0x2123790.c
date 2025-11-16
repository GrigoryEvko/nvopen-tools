// Function: sub_2123790
// Address: 0x2123790
//
__int64 *__fastcall sub_2123790(__int64 *a1, __int64 *a2, unsigned int a3)
{
  __int64 *v3; // rbx
  unsigned __int8 *v4; // rax
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 *v12; // r9
  unsigned __int64 v13; // r10
  __int128 *v14; // r14
  __int64 v15; // r11
  const void **v16; // r8
  __int64 v17; // rcx
  __int128 v19; // [rsp-30h] [rbp-B0h]
  __int128 v20; // [rsp-20h] [rbp-A0h]
  __int64 v21; // [rsp+8h] [rbp-78h]
  unsigned __int64 v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  const void **v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 *v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  int v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h]

  v3 = a2;
  v4 = (unsigned __int8 *)(a2[5] + 16LL * a3);
  v5 = *v4;
  v25 = *((_QWORD *)v4 + 1);
  sub_1F40D10((__int64)&v27, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v5, v25);
  if ( (_BYTE)v5 != (_BYTE)v28 )
    goto LABEL_2;
  if ( v25 == v29 )
  {
    if ( (_BYTE)v5 )
    {
LABEL_9:
      if ( *(_QWORD *)(*a1 + 8 * v5 + 120) )
        return v3;
    }
  }
  else if ( (_BYTE)v5 )
  {
    goto LABEL_9;
  }
LABEL_2:
  v6 = sub_2120330((__int64)a1, *(_QWORD *)(a2[4] + 80), *(_QWORD *)(a2[4] + 88));
  v8 = v7;
  v9 = sub_2120330((__int64)a1, *(_QWORD *)(a2[4] + 120), *(_QWORD *)(a2[4] + 128));
  v11 = a2[9];
  v12 = (__int64 *)a1[1];
  v13 = v9;
  v14 = (__int128 *)v3[4];
  v15 = v10;
  v16 = *(const void ***)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8 + 8);
  v17 = *(unsigned __int8 *)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8);
  v27 = v11;
  if ( v11 )
  {
    v21 = v17;
    v22 = v9;
    v23 = v10;
    v24 = v16;
    v26 = v12;
    sub_1623A60((__int64)&v27, v11, 2);
    v17 = v21;
    v13 = v22;
    v15 = v23;
    v16 = v24;
    v12 = v26;
  }
  v28 = *((_DWORD *)v3 + 16);
  *((_QWORD *)&v20 + 1) = v15;
  *(_QWORD *)&v20 = v13;
  *((_QWORD *)&v19 + 1) = v8;
  *(_QWORD *)&v19 = v6;
  v3 = sub_1D36A20(
         v12,
         136,
         (__int64)&v27,
         v17,
         v16,
         (__int64)v12,
         *v14,
         *(__int128 *)((char *)v14 + 40),
         v19,
         v20,
         v14[10]);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v3;
}
