// Function: sub_303AC30
// Address: 0x303ac30
//
__int64 __fastcall sub_303AC30(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5, int a6)
{
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned __int16 v9; // r13
  __int64 v10; // rbx
  int v11; // eax
  const __m128i *v12; // roff
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int16 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  char v24; // al
  __int128 v25; // [rsp-10h] [rbp-C0h]
  char v26; // [rsp+17h] [rbp-99h]
  __int128 v28; // [rsp+20h] [rbp-90h]
  __int64 v29; // [rsp+30h] [rbp-80h] BYREF
  int v30; // [rsp+38h] [rbp-78h]
  unsigned __int16 v31; // [rsp+40h] [rbp-70h] BYREF
  __int64 v32; // [rsp+48h] [rbp-68h]
  unsigned __int16 v33; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+58h] [rbp-58h]
  __int64 v35; // [rsp+60h] [rbp-50h]
  __int64 v36; // [rsp+68h] [rbp-48h]
  __int64 v37; // [rsp+70h] [rbp-40h]
  __int64 v38; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(_WORD *)v7;
  v10 = *(_QWORD *)(v7 + 8);
  v29 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v29, v8, 1);
  v11 = *(_DWORD *)(a2 + 72);
  v33 = v9;
  v34 = v10;
  v30 = v11;
  v12 = *(const __m128i **)(a2 + 40);
  v13 = v12[2].m128i_i64[1];
  v14 = v12[3].m128i_i64[0];
  v28 = (__int128)_mm_loadu_si128(v12);
  v15 = *(_QWORD *)(v13 + 48) + 16LL * v12[3].m128i_u32[0];
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v31 = v16;
  v32 = v17;
  if ( v16 == v9 )
  {
    if ( v9 || v17 == v10 )
      goto LABEL_5;
LABEL_10:
    v37 = sub_3007260((__int64)&v33);
    v20 = v37;
    v38 = v21;
    v26 = v21;
    if ( !v16 )
      goto LABEL_11;
    goto LABEL_16;
  }
  if ( !v9 )
    goto LABEL_10;
  if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
LABEL_23:
    BUG();
  v20 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
  v26 = byte_444C4A0[16 * v9 - 8];
  if ( !v16 )
  {
LABEL_11:
    v22 = sub_3007260((__int64)&v31);
    v24 = v23;
    v36 = v23;
    v35 = v22;
    if ( v22 != v20 )
      goto LABEL_12;
    goto LABEL_19;
  }
LABEL_16:
  if ( v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
    goto LABEL_23;
  v24 = byte_444C4A0[16 * v16 - 8];
  if ( *(_QWORD *)&byte_444C4A0[16 * v16 - 16] != v20 )
    goto LABEL_12;
LABEL_19:
  if ( v24 == v26 )
  {
LABEL_5:
    *((_QWORD *)&v25 + 1) = v14;
    *(_QWORD *)&v25 = v13;
    v18 = sub_3406EB0(a4, 540, (unsigned int)&v29, v9, v10, a6, v28, v25);
    goto LABEL_6;
  }
LABEL_12:
  v18 = 0;
LABEL_6:
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  return v18;
}
