// Function: sub_328B360
// Address: 0x328b360
//
__int64 __fastcall sub_328B360(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rcx
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  unsigned int v10; // r14d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  int v21; // r9d
  __int64 v22; // rbx
  __int128 v23; // rax
  int v24; // r9d
  __int128 v25; // rax
  int v26; // r9d
  __int64 v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+18h] [rbp-58h]
  unsigned __int16 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  int v32; // [rsp+28h] [rbp-48h]
  _QWORD v33[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(__int64 **)(a2 + 40);
  v5 = *v4;
  v6 = v4[1];
  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v29 = v6;
  v9 = *((_QWORD *)v7 + 1);
  v10 = *v7;
  v27 = v5;
  v31 = v8;
  v28 = v9;
  if ( v8 )
    sub_B96E90((__int64)&v31, v8, 1);
  v11 = *a1;
  v32 = *(_DWORD *)(a2 + 72);
  v33[0] = v5;
  v33[1] = v29;
  v12 = sub_3402EA0(v11, 189, (unsigned int)&v31, v10, v28, 0, (__int64)v33, 1);
  if ( v12 )
  {
    v5 = v12;
  }
  else if ( *(_DWORD *)(v5 + 24) != 189 && !(unsigned __int8)sub_33DD2A0(*a1, v5, v29, 0) )
  {
    v5 = sub_328AAB0(a1, a2, (__int64)&v31);
    if ( !v5 )
    {
      if ( *(_DWORD *)(v27 + 24) != 222 )
        goto LABEL_13;
      v14 = a1[1];
      v15 = *(__int64 (**)())(*(_QWORD *)v14 + 1392LL);
      if ( v15 == sub_2FE3480 )
        goto LABEL_13;
      v16 = *(_QWORD *)(*(_QWORD *)(v27 + 40) + 40LL);
      v17 = *(_QWORD *)(v16 + 104);
      v18 = *(unsigned __int16 *)(v16 + 96);
      v30 = *(_WORD *)(v16 + 96);
      if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64))v15)(v14, v10, v28, v18, v17)
        && (v19 = a1[1], v20 = *(__int64 (**)())(*(_QWORD *)v19 + 1432LL), v20 != sub_2FE34A0)
        && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v20)(
             v19,
             (unsigned int)v18,
             v17,
             v10,
             v28)
        && (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1[1] + 2192LL))(
             a1[1],
             189,
             (unsigned int)v18,
             v17)
        && (unsigned __int8)sub_328A020(a1[1], 0xBDu, v30, v17, *((unsigned __int8 *)a1 + 33)) )
      {
        v22 = *a1;
        *(_QWORD *)&v23 = sub_33FAF80(v22, 216, (unsigned int)&v31, v18, v17, v21, *(_OWORD *)*(_QWORD *)(v27 + 40));
        *(_QWORD *)&v25 = sub_33FAF80(v22, 189, (unsigned int)&v31, v18, v17, v24, v23);
        v5 = sub_33FAF80(v22, 214, (unsigned int)&v31, v10, v28, v26, v25);
      }
      else
      {
LABEL_13:
        v5 = 0;
      }
    }
  }
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v5;
}
