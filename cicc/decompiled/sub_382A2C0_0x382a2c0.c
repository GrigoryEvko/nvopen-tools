// Function: sub_382A2C0
// Address: 0x382a2c0
//
unsigned __int8 *__fastcall sub_382A2C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int16 *v3; // rax
  __int64 v4; // r9
  __int64 v5; // rdx
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // r10
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int32 v15; // eax
  __int64 v16; // rdx
  unsigned __int16 v17; // r14
  __m128i v18; // xmm0
  _DWORD *v19; // r12
  unsigned __int16 v20; // cx
  unsigned int v21; // eax
  char v22; // bl
  __int64 v23; // rsi
  unsigned __int64 v24; // r8
  unsigned int v25; // r15d
  unsigned __int64 v26; // r13
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int8 **v29; // rax
  __int64 v30; // rax
  unsigned int *v31; // rax
  __int64 v32; // rsi
  unsigned __int8 *v33; // rbx
  __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int16 v36; // cx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  char v40; // r11
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rax
  unsigned int v48; // edx
  _QWORD *v49; // rdx
  unsigned __int8 *v50; // r12
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int128 v55; // [rsp-10h] [rbp-190h]
  unsigned int v56; // [rsp+14h] [rbp-16Ch]
  unsigned __int64 v57; // [rsp+28h] [rbp-158h]
  char v58; // [rsp+30h] [rbp-150h]
  __int64 v60; // [rsp+40h] [rbp-140h]
  unsigned __int16 v61; // [rsp+48h] [rbp-138h]
  unsigned int v62; // [rsp+4Ch] [rbp-134h]
  __int64 v63; // [rsp+50h] [rbp-130h]
  __m128i v65; // [rsp+60h] [rbp-120h] BYREF
  __int64 v66; // [rsp+70h] [rbp-110h] BYREF
  int v67; // [rsp+78h] [rbp-108h]
  unsigned __int16 v68; // [rsp+80h] [rbp-100h] BYREF
  __int64 v69; // [rsp+88h] [rbp-F8h]
  unsigned __int16 v70; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v71; // [rsp+98h] [rbp-E8h]
  __int64 v72; // [rsp+A0h] [rbp-E0h]
  __int64 v73; // [rsp+A8h] [rbp-D8h]
  __int64 v74; // [rsp+B0h] [rbp-D0h]
  __int64 v75; // [rsp+B8h] [rbp-C8h]
  __m128i v76; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD v77[22]; // [rsp+D0h] [rbp-B0h] BYREF

  v3 = *(__int16 **)(a2 + 48);
  v4 = *a1;
  v5 = a1[1];
  v6 = *v3;
  v7 = *((_QWORD *)v3 + 1);
  v8 = *(_QWORD *)(v5 + 64);
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v9 == sub_2D56A50 )
  {
    v10 = v6;
    v11 = *a1;
    sub_2FE6CC0((__int64)&v76, v4, *(_QWORD *)(v5 + 64), v10, v7);
    LOWORD(v15) = v76.m128i_i16[4];
    v16 = v77[0];
    v65.m128i_i16[0] = v76.m128i_i16[4];
    v65.m128i_i64[1] = v77[0];
  }
  else
  {
    v54 = v6;
    v11 = v8;
    v15 = v9(v4, v8, v54, v7);
    v65.m128i_i32[0] = v15;
    v65.m128i_i64[1] = v16;
  }
  v62 = *(_DWORD *)(a2 + 64);
  if ( (_WORD)v15 )
  {
    v63 = 0;
    v17 = word_4456580[(unsigned __int16)v15 - 1];
  }
  else
  {
    v52 = sub_3009970((__int64)&v65, v11, v16, v12, v13);
    v63 = v53;
    v2 = v52;
    v17 = v52;
  }
  v18 = _mm_load_si128(&v65);
  LOWORD(v2) = v17;
  v60 = v2;
  v19 = (_DWORD *)*a1;
  v76 = v18;
  if ( v65.m128i_i16[0] )
  {
    v20 = v65.m128i_i16[0] - 17;
    if ( (unsigned __int16)(v65.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v65.m128i_i16[0] - 126) > 0x31u )
    {
      if ( v20 <= 0xD3u )
      {
LABEL_9:
        v21 = v19[17];
        goto LABEL_13;
      }
LABEL_12:
      v21 = v19[15];
      goto LABEL_13;
    }
    if ( v20 <= 0xD3u )
      goto LABEL_9;
  }
  else
  {
    v22 = sub_3007030((__int64)&v76);
    if ( sub_30070B0((__int64)&v76) )
      goto LABEL_9;
    if ( !v22 )
      goto LABEL_12;
  }
  v21 = v19[16];
LABEL_13:
  if ( v21 > 2 )
LABEL_55:
    BUG();
  v56 = 215 - v21;
  v23 = *(_QWORD *)(a2 + 80);
  v66 = v23;
  if ( v23 )
    sub_B96E90((__int64)&v66, v23, 1);
  v67 = *(_DWORD *)(a2 + 72);
  v76.m128i_i64[0] = (__int64)v77;
  v76.m128i_i64[1] = 0x800000000LL;
  if ( v62 > 8 )
  {
    sub_C8D5F0((__int64)&v76, v77, v62, 0x10u, v13, v14);
    goto LABEL_18;
  }
  v24 = v62;
  if ( v62 )
  {
LABEL_18:
    v25 = 0;
    while ( 1 )
    {
      v31 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * v25);
      v32 = v31[2];
      v33 = *(unsigned __int8 **)v31;
      v34 = *((_QWORD *)v31 + 1);
      v35 = *(_QWORD *)(*(_QWORD *)v31 + 48LL) + 16 * v32;
      v36 = *(_WORD *)v35;
      v37 = *(_QWORD *)(v35 + 8);
      v68 = v36;
      v69 = v37;
      if ( v17 != v36 )
        break;
      if ( !v17 && v37 != v63 )
      {
        v70 = 0;
        v71 = v63;
LABEL_25:
        v61 = v36;
        v38 = sub_3007260((__int64)&v70);
        v36 = v61;
        v14 = v39;
        v74 = v38;
        v24 = v38;
        v75 = v39;
        v40 = v39;
LABEL_26:
        if ( v36 )
        {
          if ( v36 == 1 || (unsigned __int16)(v36 - 504) <= 7u )
            goto LABEL_55;
          v45 = *(_QWORD *)&byte_444C4A0[16 * v36 - 16];
          LOBYTE(v44) = byte_444C4A0[16 * v36 - 8];
        }
        else
        {
          v57 = v24;
          v58 = v40;
          v41 = sub_3007260((__int64)&v68);
          v24 = v57;
          v42 = v41;
          v44 = v43;
          v40 = v58;
          v36 = 0;
          v72 = v42;
          v45 = v42;
          v73 = v44;
        }
        if ( (!(_BYTE)v44 || v40) && v45 < v24 )
        {
          v46 = 215;
          if ( v36 == 2 && *((_DWORD *)v33 + 6) == 11 )
            v46 = v56;
          v47 = v60;
          LOWORD(v47) = v17;
          v60 = v47;
          v33 = sub_33FAF80(a1[1], v46, (__int64)&v66, (unsigned int)v47, v63, v14, v18);
          v32 = v48;
        }
      }
      v26 = v32 | v34 & 0xFFFFFFFF00000000LL;
      v27 = v76.m128i_u32[2];
      v28 = v76.m128i_u32[2] + 1LL;
      if ( v28 > v76.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v76, v77, v28, 0x10u, v24, v14);
        v27 = v76.m128i_u32[2];
      }
      v29 = (unsigned __int8 **)(v76.m128i_i64[0] + 16 * v27);
      ++v25;
      *v29 = v33;
      v29[1] = (unsigned __int8 *)v26;
      v30 = (unsigned int)++v76.m128i_i32[2];
      if ( v25 == v62 )
      {
        v49 = (_QWORD *)v76.m128i_i64[0];
        goto LABEL_44;
      }
    }
    v70 = v17;
    v71 = v63;
    if ( !v17 )
      goto LABEL_25;
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      goto LABEL_55;
    v24 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
    v40 = byte_444C4A0[16 * v17 - 8];
    goto LABEL_26;
  }
  v49 = v77;
  v30 = 0;
LABEL_44:
  *((_QWORD *)&v55 + 1) = v30;
  *(_QWORD *)&v55 = v49;
  v50 = sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v66, v65.m128i_i64[0], v65.m128i_i64[1], a1[1], v55);
  if ( (_QWORD *)v76.m128i_i64[0] != v77 )
    _libc_free(v76.m128i_u64[0]);
  if ( v66 )
    sub_B91220((__int64)&v66, v66);
  return v50;
}
