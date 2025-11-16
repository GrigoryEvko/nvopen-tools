// Function: sub_382AA40
// Address: 0x382aa40
//
unsigned __int8 *__fastcall sub_382AA40(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 v6; // rdx
  __int16 *v7; // rax
  __int64 v8; // r10
  unsigned __int16 v9; // si
  __int64 v10; // r8
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  unsigned __int16 v14; // bx
  _QWORD *v15; // r13
  _DWORD *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r9
  unsigned __int8 *v22; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-70h] BYREF
  int v30; // [rsp+8h] [rbp-68h]
  unsigned int v31; // [rsp+10h] [rbp-60h] BYREF
  __int64 v32; // [rsp+18h] [rbp-58h]
  unsigned __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h] BYREF
  __int64 v36; // [rsp+38h] [rbp-38h]
  __int64 v37; // [rsp+40h] [rbp-30h]

  v4 = *(_QWORD *)(a2 + 80);
  v29 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v29, v4, 1);
  v5 = *a1;
  v6 = a1[1];
  v30 = *(_DWORD *)(a2 + 72);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *(_QWORD *)(v6 + 64);
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    v12 = v9;
    v13 = v5;
    sub_2FE6CC0((__int64)&v35, v5, *(_QWORD *)(v6 + 64), v12, v10);
    v14 = v36;
    LOWORD(v31) = v36;
    v32 = v37;
  }
  else
  {
    v27 = v9;
    v13 = v8;
    v31 = v11(v5, v8, v27, v10);
    v14 = v31;
    v32 = v28;
  }
  v15 = (_QWORD *)a1[1];
  v16 = (_DWORD *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 96LL) + 24LL);
  if ( v14 )
  {
    if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
    {
      v34 = 0;
      v14 = word_4456580[v14 - 1];
      LOWORD(v33) = v14;
      if ( !v14 )
        goto LABEL_9;
      goto LABEL_17;
    }
    goto LABEL_7;
  }
  if ( !sub_30070B0((__int64)&v31) )
  {
LABEL_7:
    v17 = v32;
    goto LABEL_8;
  }
  v14 = sub_3009970((__int64)&v31, v13, v24, v25, v26);
LABEL_8:
  LOWORD(v33) = v14;
  v34 = v17;
  if ( !v14 )
  {
LABEL_9:
    v18 = sub_3007260((__int64)&v33);
    v19 = v20;
    v35 = v18;
    LODWORD(v20) = v18;
    v36 = v19;
    goto LABEL_10;
  }
LABEL_17:
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    BUG();
  v20 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
LABEL_10:
  sub_C44830((__int64)&v33, v16, v20);
  v22 = sub_3402600(v15, (unsigned __int64 *)&v29, v31, v32, (__int64)&v33, v21, a3);
  if ( (unsigned int)v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  return v22;
}
