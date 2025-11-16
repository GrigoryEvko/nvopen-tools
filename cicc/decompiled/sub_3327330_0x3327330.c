// Function: sub_3327330
// Address: 0x3327330
//
__int64 __fastcall sub_3327330(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  __m128i v5; // xmm1
  unsigned __int16 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rax
  int v10; // ebx
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // rbx
  int v14; // r14d
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  __int128 v18; // rax
  int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // r9d
  __int64 v23; // r14
  __int128 v24; // rax
  int v26; // [rsp+8h] [rbp-118h]
  __int128 v27; // [rsp+10h] [rbp-110h]
  __int128 v28; // [rsp+10h] [rbp-110h]
  int v29; // [rsp+10h] [rbp-110h]
  __int64 v30; // [rsp+20h] [rbp-100h] BYREF
  int v31; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v32; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v34; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-D8h]
  int v36; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+58h] [rbp-C8h]
  __int64 v38; // [rsp+60h] [rbp-C0h]
  int v39; // [rsp+68h] [rbp-B8h]
  __int64 v40; // [rsp+70h] [rbp-B0h]
  int v41; // [rsp+78h] [rbp-A8h]
  __int64 v42; // [rsp+80h] [rbp-A0h]
  int v43; // [rsp+88h] [rbp-98h]
  __int64 v44; // [rsp+90h] [rbp-90h]
  __int64 v45; // [rsp+98h] [rbp-88h]
  int v46; // [rsp+A0h] [rbp-80h]
  char v47; // [rsp+A4h] [rbp-7Ch]
  __int64 v48; // [rsp+A8h] [rbp-78h]
  __int64 v49; // [rsp+B0h] [rbp-70h]
  int v50; // [rsp+B8h] [rbp-68h]
  char v51; // [rsp+BCh] [rbp-64h]
  __int128 v52; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v53; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+D8h] [rbp-48h]

  v3 = *(_QWORD *)(a2 + 80);
  v30 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v30, v3, 1);
  v4 = *(_QWORD *)(a1 + 8);
  v31 = *(_DWORD *)(a2 + 72);
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v6 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = 1;
  v10 = (unsigned __int16)v7;
  if ( (_WORD)v7 == 1 || (_WORD)v7 && (v9 = (unsigned __int16)v7, *(_QWORD *)(v4 + 8 * v7 + 112)) )
  {
    if ( (*(_BYTE *)(v4 + 500 * v9 + 6566) & 0xFB) == 0 )
    {
      v29 = v8;
      *(_QWORD *)&v24 = sub_33FE730(*(_QWORD *)(a1 + 16), &v30, (unsigned __int16)v7, v8, 0, 0.0);
      v23 = sub_3406EB0(*(_QWORD *)(a1 + 16), 152, (unsigned int)&v30, v10, v29, v29, *(_OWORD *)&v5, v24);
      goto LABEL_22;
    }
  }
  LOWORD(v36) = 0;
  v37 = 0;
  HIWORD(v14) = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  *(_QWORD *)&v52 = 0;
  DWORD2(v52) = 0;
  v54 = 1;
  v53 = 0;
  sub_3326BF0(a1, (__int64)&v36, (int)&v30, v5.m128i_i64[0], v5.m128i_i64[1], v8);
  v11 = *(_QWORD *)(a1 + 16);
  v12 = *(_QWORD *)(v52 + 48) + 16LL * DWORD2(v52);
  v13 = *(_QWORD *)(v12 + 8);
  LOWORD(v14) = *(_WORD *)v12;
  v15 = v54;
  v33 = v54;
  if ( v54 <= 0x40 )
  {
    v16 = v53;
LABEL_8:
    v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & ~v16;
    if ( !v15 )
      v17 = 0;
    v32 = v17;
    goto LABEL_11;
  }
  v26 = v11;
  sub_C43780((__int64)&v32, (const void **)&v53);
  v15 = v33;
  LODWORD(v11) = v26;
  if ( v33 <= 0x40 )
  {
    v16 = v32;
    goto LABEL_8;
  }
  sub_C43D10((__int64)&v32);
  v15 = v33;
  v17 = v32;
  LODWORD(v11) = v26;
LABEL_11:
  v34 = v17;
  v35 = v15;
  v33 = 0;
  *(_QWORD *)&v18 = sub_34007B0(v11, (unsigned int)&v34, (unsigned int)&v30, v14, v13, 0, 0);
  if ( v35 > 0x40 && v34 )
  {
    v27 = v18;
    j_j___libc_free_0_0(v34);
    v18 = v27;
  }
  if ( v33 > 0x40 && v32 )
  {
    v28 = v18;
    j_j___libc_free_0_0(v32);
    v18 = v28;
  }
  v20 = sub_3406EB0(*(_QWORD *)(a1 + 16), 186, (unsigned int)&v30, v14, v13, v19, v52, v18);
  v23 = sub_3325820(a1, &v36, (int)&v30, v20, v21, v22);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
LABEL_22:
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v23;
}
