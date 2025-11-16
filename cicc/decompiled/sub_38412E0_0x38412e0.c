// Function: sub_38412E0
// Address: 0x38412e0
//
void *__fastcall sub_38412E0(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // r9
  __int64 v6; // rsi
  __int64 v7; // r8
  const __m128i *v8; // rbx
  unsigned __int64 v9; // r10
  __int64 v10; // r11
  __int64 v11; // rax
  int v12; // ecx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  const __m128i *v15; // r13
  __m128 *v16; // rdx
  unsigned __int8 *v17; // rsi
  _BYTE *v18; // r8
  unsigned int v19; // edx
  __int16 *v20; // rax
  __int16 v21; // r13
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned __int16 v24; // bx
  __int64 v25; // r8
  bool v26; // al
  __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // rsi
  unsigned __int8 *v34; // rax
  int v35; // edx
  int v36; // ebx
  unsigned __int8 *v37; // rdx
  unsigned __int64 v38; // rax
  int v39; // r9d
  void *v40; // r12
  __int64 v41; // rsi
  _QWORD *v42; // rbx
  unsigned __int64 v43; // r13
  __int64 v44; // r14
  __int64 v45; // rsi
  unsigned __int16 *v47; // rax
  unsigned __int8 *v48; // rax
  int v49; // edx
  int v50; // esi
  __int64 v51; // rdx
  __int64 v52; // rcx
  unsigned __int16 v53; // ax
  __int64 v54; // rdx
  __int128 v55; // [rsp-20h] [rbp-140h]
  __int128 v56; // [rsp-10h] [rbp-130h]
  __int64 v57; // [rsp+8h] [rbp-118h]
  unsigned __int64 v58; // [rsp+10h] [rbp-110h]
  __int64 v59; // [rsp+18h] [rbp-108h]
  int v60; // [rsp+18h] [rbp-108h]
  __int64 v61; // [rsp+20h] [rbp-100h]
  __int64 v62; // [rsp+20h] [rbp-100h]
  __int64 v63; // [rsp+20h] [rbp-100h]
  __int64 v65; // [rsp+28h] [rbp-F8h]
  __int64 v66; // [rsp+28h] [rbp-F8h]
  int v67; // [rsp+28h] [rbp-F8h]
  __int64 v68; // [rsp+50h] [rbp-D0h] BYREF
  int v69; // [rsp+58h] [rbp-C8h]
  unsigned int v70; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+68h] [rbp-B8h]
  unsigned __int16 v72; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v73; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v74; // [rsp+80h] [rbp-A0h]
  __int64 v75; // [rsp+88h] [rbp-98h]
  unsigned __int64 v76; // [rsp+90h] [rbp-90h] BYREF
  __int64 v77; // [rsp+98h] [rbp-88h]
  _BYTE *v78; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v79; // [rsp+A8h] [rbp-78h]
  _BYTE v80[112]; // [rsp+B0h] [rbp-70h] BYREF

  v4 = a3;
  v6 = *(_QWORD *)(a2 + 80);
  v68 = v6;
  if ( v6 )
  {
    sub_B96E90((__int64)&v68, v6, 1);
    v4 = a3;
  }
  v7 = (unsigned int)v4;
  v8 = *(const __m128i **)(a2 + 40);
  v69 = *(_DWORD *)(a2 + 72);
  v9 = v8->m128i_u64[5 * (unsigned int)v4];
  v10 = v8->m128i_i64[5 * (unsigned int)v4 + 1];
  v78 = v80;
  v11 = *(unsigned int *)(a2 + 64);
  v79 = 0x400000000LL;
  v12 = 0;
  v11 *= 5;
  v13 = 8 * v11;
  v14 = 0xCCCCCCCCCCCCCCCDLL * v11;
  v15 = (const __m128i *)((char *)v8 + v13);
  v16 = (__m128 *)v80;
  if ( v13 > 0xA0 )
  {
    v57 = v10;
    v58 = v9;
    v60 = v4;
    v63 = (unsigned int)v4;
    v67 = v14;
    sub_C8D5F0((__int64)&v78, v80, v14, 0x10u, (unsigned int)v4, v4);
    v12 = v79;
    v10 = v57;
    v9 = v58;
    LODWORD(v4) = v60;
    v7 = v63;
    LODWORD(v14) = v67;
    v16 = (__m128 *)&v78[16 * (unsigned int)v79];
  }
  if ( v8 != v15 )
  {
    do
    {
      if ( v16 )
      {
        a4 = _mm_loadu_si128(v8);
        *v16 = (__m128)a4;
      }
      v8 = (const __m128i *)((char *)v8 + 40);
      ++v16;
    }
    while ( v15 != v8 );
    v12 = v79;
  }
  LODWORD(v79) = v12 + v14;
  if ( (_DWORD)v4 == 2 )
  {
    v47 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL));
    v48 = sub_375B580(a1, v9, a4, v10, *v47, *((_QWORD *)v47 + 1));
    v50 = v49;
    v51 = (__int64)v78;
    *((_QWORD *)v78 + 4) = v48;
    v52 = (unsigned int)v79;
    *(_DWORD *)(v51 + 40) = v50;
    v40 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, v51, v52);
    goto LABEL_29;
  }
  v65 = v7;
  v17 = sub_3841260(a1, a2, v9, v10, a4);
  v18 = &v78[16 * v65];
  *(_QWORD *)v18 = v17;
  *((_DWORD *)v18 + 2) = v19;
  v20 = *(__int16 **)(a2 + 48);
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  v23 = *((_QWORD *)v17 + 6) + 16LL * v19;
  v24 = *(_WORD *)v23;
  v25 = *(_QWORD *)(v23 + 8);
  LOWORD(v70) = v21;
  v71 = v22;
  LOWORD(v76) = v24;
  v77 = v25;
  if ( v24 )
  {
    if ( (unsigned __int16)(v24 - 17) > 0xD3u )
    {
      v28 = v24;
      if ( v24 != v21 )
        goto LABEL_14;
      goto LABEL_25;
    }
    v25 = 0;
    v24 = word_4456580[v24 - 1];
  }
  else
  {
    v59 = v22;
    v61 = v25;
    v26 = sub_30070B0((__int64)&v76);
    v25 = v61;
    v22 = v59;
    if ( !v26 )
    {
      v28 = 0;
      if ( v21 )
        goto LABEL_14;
LABEL_44:
      if ( v22 != v25 )
      {
        v24 = 0;
LABEL_14:
        v62 = v28;
        v73 = v25;
        v66 = v25;
        v72 = v24;
        v76 = sub_2D5B750(&v72);
        v77 = v29;
        v30 = sub_2D5B750((unsigned __int16 *)&v70);
        v75 = v31;
        v74 = v30;
        if ( !(_BYTE)v31 && (_BYTE)v77 || v74 < v76 )
        {
          v32 = *(_DWORD *)(a2 + 24);
          if ( v32 <= 390 )
          {
            if ( v32 <= 388 )
            {
              if ( v32 <= 386 )
              {
                if ( v32 > 381 )
                  goto LABEL_21;
                goto LABEL_48;
              }
LABEL_46:
              v33 = 213;
              goto LABEL_22;
            }
          }
          else
          {
            if ( v32 <= 477 )
            {
              if ( v32 <= 475 )
              {
                if ( (unsigned int)(v32 - 471) <= 4 )
                {
LABEL_21:
                  v33 = 215;
LABEL_22:
                  v34 = sub_33FAF80(*(_QWORD *)(a1 + 8), v33, (__int64)&v68, (unsigned int)v62, v66, v62, a4);
                  v36 = v35;
                  v37 = v34;
                  v38 = (unsigned __int64)v78;
                  *(_QWORD *)v78 = v37;
                  *(_DWORD *)(v38 + 8) = v36;
                  *((_QWORD *)&v55 + 1) = (unsigned int)v79;
                  *(_QWORD *)&v55 = v78;
                  sub_33FC220(
                    *(_QWORD **)(a1 + 8),
                    *(unsigned int *)(a2 + 24),
                    (__int64)&v68,
                    (unsigned int)v62,
                    v66,
                    v62,
                    v55);
                  v40 = sub_33FAF80(*(_QWORD *)(a1 + 8), 216, (__int64)&v68, v70, v71, v39, a4);
                  goto LABEL_29;
                }
LABEL_48:
                BUG();
              }
              goto LABEL_46;
            }
            if ( (unsigned int)(v32 - 478) > 1 )
              goto LABEL_48;
          }
          v33 = 214;
          goto LABEL_22;
        }
        goto LABEL_25;
      }
      goto LABEL_25;
    }
    v53 = sub_3009970((__int64)&v76, (__int64)v17, v27, v59, v61);
    v22 = v59;
    v24 = v53;
    v25 = v54;
  }
  v28 = v24;
  if ( v21 != v24 )
    goto LABEL_14;
  if ( !v21 )
    goto LABEL_44;
LABEL_25:
  v41 = *(_QWORD *)(a2 + 80);
  v42 = *(_QWORD **)(a1 + 8);
  v43 = (unsigned __int64)v78;
  v44 = (unsigned int)v79;
  v76 = v41;
  if ( v41 )
    sub_B96E90((__int64)&v76, v41, 1);
  v45 = *(unsigned int *)(a2 + 24);
  *((_QWORD *)&v56 + 1) = v44;
  *(_QWORD *)&v56 = v43;
  LODWORD(v77) = *(_DWORD *)(a2 + 72);
  v40 = sub_33FC220(v42, v45, (__int64)&v76, v70, v71, (__int64)&v76, v56);
  if ( v76 )
    sub_B91220((__int64)&v76, v76);
LABEL_29:
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
  return v40;
}
