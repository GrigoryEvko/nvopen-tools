// Function: sub_38371B0
// Address: 0x38371b0
//
unsigned __int8 *__fastcall sub_38371B0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  _QWORD *v5; // r9
  __int16 v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r10
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v11; // r10
  __int64 v12; // r8
  int v13; // r13d
  char v14; // al
  __int16 v15; // r9d^2
  __int16 v16; // r10
  __int64 v17; // r11
  __int64 v18; // r8
  bool v19; // zf
  __int64 v20; // rax
  unsigned __int64 v21; // r8
  __m128i v22; // xmm0
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned __int16 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rax
  unsigned int v29; // edx
  unsigned __int8 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // r11
  _QWORD *v34; // r9
  unsigned __int16 *v35; // rdx
  unsigned __int8 *v36; // r10
  __int64 v37; // rbx
  unsigned int v38; // ecx
  unsigned int v39; // esi
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned __int8 *v42; // r12
  unsigned __int8 *v44; // rax
  int v45; // r9d
  __int64 v46; // rsi
  __int64 v47; // rbx
  unsigned int v48; // edx
  unsigned __int16 *v49; // rdx
  __int64 v50; // r8
  __int64 v51; // rcx
  __int64 v52; // rsi
  unsigned __int8 *v53; // rax
  bool v54; // al
  unsigned __int16 v55; // r9
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int16 v58; // r10
  unsigned int v59; // r9d
  __int64 v60; // r8
  __int128 v61; // [rsp-30h] [rbp-E0h]
  __int64 v62; // [rsp+0h] [rbp-B0h]
  __int64 v63; // [rsp+0h] [rbp-B0h]
  unsigned __int8 *v64; // [rsp+0h] [rbp-B0h]
  __int64 v65; // [rsp+8h] [rbp-A8h]
  unsigned int v66; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v67; // [rsp+18h] [rbp-98h]
  __int64 v68; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v69; // [rsp+18h] [rbp-98h]
  _QWORD *v70; // [rsp+18h] [rbp-98h]
  __int64 v71; // [rsp+18h] [rbp-98h]
  __int64 v72; // [rsp+18h] [rbp-98h]
  __int128 v73; // [rsp+20h] [rbp-90h]
  __int64 v74; // [rsp+20h] [rbp-90h]
  __int16 v75; // [rsp+22h] [rbp-8Eh]
  __int64 v76; // [rsp+30h] [rbp-80h]
  __int64 v77; // [rsp+30h] [rbp-80h]
  unsigned __int16 v78; // [rsp+30h] [rbp-80h]
  __int16 v79; // [rsp+30h] [rbp-80h]
  __int64 v80; // [rsp+50h] [rbp-60h] BYREF
  __int64 v81; // [rsp+58h] [rbp-58h]
  __int64 v82; // [rsp+60h] [rbp-50h] BYREF
  int v83; // [rsp+68h] [rbp-48h]
  __int64 v84; // [rsp+70h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = (_QWORD *)*a1;
  v6 = *v4;
  v7 = *((_QWORD *)v4 + 1);
  v8 = *(_QWORD *)*a1;
  v9 = *(_QWORD *)(a1[1] + 64);
  LOWORD(v80) = v6;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v8 + 592);
  v81 = v7;
  if ( v10 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v82, (__int64)v5, v9, v80, v81);
    v11 = v83;
    v12 = v84;
  }
  else
  {
    v55 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))v10)(v5, v9, (unsigned int)v80);
    v12 = v56;
    v11 = v55;
  }
  v13 = *(_DWORD *)(a2 + 24);
  if ( v13 == 200 )
  {
    if ( v6 )
    {
      if ( (unsigned __int16)(v6 - 17) <= 0xD3u )
        goto LABEL_9;
      v76 = v12;
      if ( !v11 )
        goto LABEL_9;
    }
    else
    {
      v71 = v12;
      v78 = v11;
      v54 = sub_30070B0((__int64)&v80);
      v11 = v78;
      LODWORD(v12) = v71;
      if ( v54 )
        goto LABEL_9;
      v76 = v71;
      if ( !v11 )
        goto LABEL_9;
    }
    if ( *(_QWORD *)(*a1 + 8LL * v11 + 112) )
    {
      v14 = sub_38138F0(*a1, 0xC8u, v11, 0, v12);
      v18 = v76;
      if ( !v14 )
      {
        v79 = v16;
        v72 = v18;
        v75 = v15;
        if ( sub_3456620(v17, a2, (_QWORD *)a1[1]) )
        {
          v57 = a1[1];
          v58 = v79;
          HIWORD(v59) = v75;
          v82 = *(_QWORD *)(a2 + 80);
          v60 = v72;
          if ( v82 )
          {
            sub_3813810(&v82);
            v60 = v72;
            HIWORD(v59) = v75;
            v58 = v79;
          }
          LOWORD(v59) = v58;
          v83 = *(_DWORD *)(a2 + 72);
          v42 = sub_33FAF80(v57, 215, (__int64)&v82, v59, v60, v59, a3);
          sub_9C6650(&v82);
          return v42;
        }
        v13 = *(_DWORD *)(a2 + 24);
      }
    }
  }
LABEL_9:
  v19 = !sub_33CB110(v13);
  v20 = *(_QWORD *)(a2 + 40);
  if ( v19 )
  {
    v44 = sub_37AF270((__int64)a1, *(_QWORD *)v20, *(_QWORD *)(v20 + 8), a3);
    v46 = *(_QWORD *)(a2 + 80);
    v47 = a1[1];
    v49 = (unsigned __int16 *)(*((_QWORD *)v44 + 6) + 16LL * v48);
    v50 = *((_QWORD *)v49 + 1);
    v51 = *v49;
    v82 = v46;
    if ( v46 )
    {
      v74 = v51;
      v77 = v50;
      sub_B96E90((__int64)&v82, v46, 1);
      v51 = v74;
      v50 = v77;
    }
    v52 = *(unsigned int *)(a2 + 24);
    v83 = *(_DWORD *)(a2 + 72);
    v53 = sub_33FAF80(v47, v52, (__int64)&v82, v51, v50, v45, a3);
    v41 = v82;
    v42 = v53;
    if ( v82 )
      goto LABEL_17;
  }
  else
  {
    v21 = *(_QWORD *)v20;
    v22 = _mm_loadu_si128((const __m128i *)(v20 + 40));
    v23 = *(_QWORD *)(v20 + 8);
    v24 = *(_QWORD *)(*(_QWORD *)v20 + 80LL);
    v73 = (__int128)_mm_loadu_si128((const __m128i *)(v20 + 80));
    v25 = *(_QWORD *)(*(_QWORD *)v20 + 48LL) + 16LL * *(unsigned int *)(v20 + 8);
    v26 = *(_WORD *)v25;
    v27 = *(_QWORD *)(v25 + 8);
    v82 = v24;
    if ( v24 )
    {
      v62 = v23;
      v67 = v21;
      sub_B96E90((__int64)&v82, v24, 1);
      v23 = v62;
      v21 = v67;
    }
    v68 = v23;
    v83 = *(_DWORD *)(v21 + 72);
    v28 = sub_37AE0F0((__int64)a1, v21, v23);
    v30 = sub_3400810(
            (_QWORD *)a1[1],
            v28,
            v68 & 0xFFFFFFFF00000000LL | v29,
            v22.m128i_i64[0],
            v22.m128i_i64[1],
            (__int64)&v82,
            v22,
            v73,
            v26,
            v27);
    if ( v82 )
    {
      v63 = v31;
      v69 = v30;
      sub_B91220((__int64)&v82, v82);
      v31 = v63;
      v30 = v69;
    }
    v32 = *(_QWORD *)(a2 + 80);
    v33 = v31;
    v34 = (_QWORD *)a1[1];
    v35 = (unsigned __int16 *)(*((_QWORD *)v30 + 6) + 16LL * (unsigned int)v31);
    v36 = v30;
    v37 = *((_QWORD *)v35 + 1);
    v38 = *v35;
    v82 = v32;
    if ( v32 )
    {
      v66 = v38;
      v65 = v33;
      v70 = v34;
      v64 = v30;
      sub_B96E90((__int64)&v82, v32, 1);
      v38 = v66;
      v36 = v64;
      v33 = v65;
      v34 = v70;
    }
    v39 = *(_DWORD *)(a2 + 24);
    *((_QWORD *)&v61 + 1) = v33;
    *(_QWORD *)&v61 = v36;
    v83 = *(_DWORD *)(a2 + 72);
    v40 = sub_340F900(v34, v39, (__int64)&v82, v38, v37, (__int64)v34, v61, *(_OWORD *)&v22, v73);
    v41 = v82;
    v42 = (unsigned __int8 *)v40;
    if ( v82 )
LABEL_17:
      sub_B91220((__int64)&v82, v41);
  }
  return v42;
}
