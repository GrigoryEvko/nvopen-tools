// Function: sub_36F44B0
// Address: 0x36f44b0
//
__int64 __fastcall sub_36F44B0(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r15d
  int v4; // eax
  __int64 v5; // r15
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // esi
  __int64 v10; // rdi
  __int64 (*v11)(void); // rax
  __int64 v12; // r12
  __int64 v13; // r13
  int v14; // eax
  __int64 v15; // rax
  __int64 v17; // rbx
  const __m128i *v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // r14
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rbx
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 *v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  unsigned __int8 *v36; // rsi
  __int32 v37; // ebx
  __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // r14
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rbx
  __int64 v48; // rdi
  __int64 (__fastcall *v49)(__int64); // rax
  __int64 v50; // rdi
  __int64 v51; // rbx
  __int64 v52; // r12
  __int64 v53; // r14
  __int64 v54; // rbx
  unsigned __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // r14
  __int64 (*v58)(void); // rax
  __int64 v59; // r13
  unsigned __int8 *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // r13
  _QWORD *v63; // rax
  __int64 v64; // rbx
  __int32 v65; // eax
  __int64 v66; // rdx
  __int64 v67; // rax
  int v68; // eax
  unsigned __int64 v69; // rdi
  _QWORD *v70; // [rsp+8h] [rbp-118h]
  _QWORD *v72; // [rsp+20h] [rbp-100h]
  bool v73; // [rsp+2Ch] [rbp-F4h]
  __int32 v74; // [rsp+2Ch] [rbp-F4h]
  __int64 v75; // [rsp+30h] [rbp-F0h]
  __int64 v76; // [rsp+30h] [rbp-F0h]
  __int64 v77; // [rsp+30h] [rbp-F0h]
  int v78; // [rsp+38h] [rbp-E8h]
  __int32 v79; // [rsp+38h] [rbp-E8h]
  __int64 v80; // [rsp+38h] [rbp-E8h]
  __int64 v81; // [rsp+40h] [rbp-E0h]
  __int64 v82; // [rsp+40h] [rbp-E0h]
  __int64 v83; // [rsp+40h] [rbp-E0h]
  __int64 v84; // [rsp+40h] [rbp-E0h]
  _QWORD *v85; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *v86; // [rsp+68h] [rbp-B8h] BYREF
  unsigned __int8 *v87[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v88; // [rsp+90h] [rbp-90h] BYREF
  __m128i v89; // [rsp+A0h] [rbp-80h]
  __int64 v90; // [rsp+B0h] [rbp-70h]
  __m128i v91; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v92; // [rsp+D0h] [rbp-50h]
  __int64 v93; // [rsp+D8h] [rbp-48h]
  __int64 v94; // [rsp+E0h] [rbp-40h]

  v2 = sub_BB98D0(a1, *(_QWORD *)a2);
  if ( (_BYTE)v2 )
    return 0;
  v3 = v2;
  v72 = *(_QWORD **)(a2 + 328);
  v70 = (_QWORD *)(a2 + 320);
  if ( v72 == (_QWORD *)(a2 + 320) )
    goto LABEL_16;
  do
  {
    if ( (_QWORD *)v72[7] == v72 + 6 )
      goto LABEL_15;
    v4 = v3;
    v5 = v72[7];
    v6 = v4;
    do
    {
      if ( !v5 )
        BUG();
      v7 = v5;
      if ( (*(_BYTE *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 44) & 8) != 0 )
      {
        do
          v7 = *(_QWORD *)(v7 + 8);
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 );
      }
      v85 = *(_QWORD **)(v7 + 8);
      if ( (unsigned int)*(unsigned __int16 *)(v5 + 68) - 7006 <= 1 )
      {
        v8 = *(_QWORD *)(v5 + 32);
        if ( !*(_BYTE *)(v8 + 40) )
        {
          v9 = *(_DWORD *)(v8 + 48);
          if ( v9 < 0 )
          {
            v44 = *(_QWORD *)(v5 + 24);
            v45 = *(_QWORD *)(v44 + 32);
            v46 = sub_2EBEE90(*(_QWORD *)(v45 + 32), v9);
            v47 = v46;
            if ( v46 )
            {
              if ( v44 == *(_QWORD *)(v46 + 24) && (unsigned int)*(unsigned __int16 *)(v46 + 68) - 2679 <= 1 )
              {
                v48 = *(_QWORD *)(v45 + 16);
                v49 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v48 + 200LL);
                v50 = v49 == sub_3020000 ? v48 + 456 : ((__int64 (*)(void))v49)();
                v51 = *(_QWORD *)(v47 + 32);
                if ( !*(_BYTE *)(v51 + 40)
                  && *(_DWORD *)(v51 + 48) == (*(unsigned int (__fastcall **)(__int64, __int64))(*(_QWORD *)v50 + 680LL))(
                                                v50,
                                                v45) )
                {
                  v52 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 32LL);
                  v84 = *(_QWORD *)(v5 + 24);
                  v53 = *(_QWORD *)(v52 + 32);
                  v80 = v53;
                  v54 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v52 + 16) + 128LL))(*(_QWORD *)(v52 + 16));
                  v55 = sub_2EBEE90(v53, *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL));
                  v56 = *(_QWORD *)(v52 + 16);
                  v57 = v55;
                  v58 = *(__int64 (**)(void))(*(_QWORD *)v56 + 200LL);
                  if ( (char *)v58 == (char *)sub_3020000 )
                    v77 = v56 + 456;
                  else
                    v77 = v58();
                  v59 = *(_QWORD *)(v54 + 8);
                  v60 = *(unsigned __int8 **)(v5 + 56);
                  v74 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 8LL);
                  v61 = *(unsigned __int16 *)(v57 + 68);
                  v86 = v60;
                  v62 = v59 - 40 * v61;
                  if ( v60 )
                  {
                    sub_B96E90((__int64)&v86, (__int64)v60, 1);
                    v88.m128i_i64[0] = (__int64)v86;
                    if ( v86 )
                    {
                      sub_B976B0((__int64)&v86, v86, (__int64)&v88);
                      v86 = 0;
                      v88.m128i_i64[1] = 0;
                      v89.m128i_i64[0] = 0;
                      v87[0] = (unsigned __int8 *)v88.m128i_i64[0];
                      if ( v88.m128i_i64[0] )
                        sub_B96E90((__int64)v87, v88.m128i_i64[0], 1);
                      goto LABEL_59;
                    }
                  }
                  else
                  {
                    v88.m128i_i64[0] = 0;
                  }
                  v88.m128i_i64[1] = 0;
                  v89.m128i_i64[0] = 0;
                  v87[0] = 0;
LABEL_59:
                  v63 = sub_2E7B380((_QWORD *)v52, v62, v87, 0);
                  v64 = (__int64)v63;
                  if ( v88.m128i_i64[1] )
                    sub_2E882B0((__int64)v63, v52, v88.m128i_i64[1]);
                  if ( v89.m128i_i64[0] )
                    sub_2E88680(v64, v52, v89.m128i_i64[0]);
                  v91.m128i_i64[0] = 0x10000000;
                  v92 = 0;
                  v91.m128i_i32[2] = v74;
                  v93 = 0;
                  v94 = 0;
                  sub_2E8EAD0(v64, v52, &v91);
                  if ( v87[0] )
                    sub_B91220((__int64)v87, (__int64)v87[0]);
                  v65 = sub_30590D0(v77, v52);
                  v91.m128i_i64[0] = 0;
                  v91.m128i_i32[2] = v65;
                  v92 = 0;
                  v93 = 0;
                  v94 = 0;
                  sub_2E8EAD0(v64, v52, &v91);
                  sub_2E8EAD0(v64, v52, (const __m128i *)(*(_QWORD *)(v57 + 32) + 80LL));
                  if ( v88.m128i_i64[0] )
                    sub_B91220((__int64)&v88, v88.m128i_i64[0]);
                  if ( v86 )
                    sub_B91220((__int64)&v86, (__int64)v86);
                  sub_2E31040((__int64 *)(v84 + 40), v64);
                  v66 = *(_QWORD *)v5;
                  v67 = *(_QWORD *)v64;
                  *(_QWORD *)(v64 + 8) = v5;
                  v66 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)v64 = v66 | v67 & 7;
                  *(_QWORD *)(v66 + 8) = v64;
                  *(_QWORD *)v5 = *(_QWORD *)v5 & 7LL | v64;
                  if ( (unsigned __int8)sub_2EBEF70(v80, *(_DWORD *)(*(_QWORD *)(v57 + 32) + 8LL)) )
                    sub_2E88E20(v57);
                  v6 = 1;
                  sub_2E88E20(v5);
                }
              }
            }
          }
        }
      }
      if ( (_BYTE)qword_5041028 )
      {
        if ( *(_WORD *)(v5 + 68) == 396 )
        {
          v17 = *(_QWORD *)(a2 + 32);
          v18 = *(const __m128i **)(v5 + 32);
          v88 = _mm_loadu_si128(v18);
          v89 = _mm_loadu_si128(v18 + 1);
          v90 = v18[2].m128i_i64[0];
          if ( !(v18->m128i_i8[0] | v88.m128i_i8[3] & 0x10) )
          {
            v19 = sub_2EBEE90(v17, v18->m128i_i32[2]);
            v20 = v19;
            if ( v19 )
            {
              if ( *(_WORD *)(v19 + 68) == 3545 && (*(_DWORD *)(v19 + 40) & 0xFFFFFFu) > 3 )
              {
                v21 = *(_QWORD *)(v19 + 32);
                v73 = *(_BYTE *)(v21 + 120) == 1
                   && *(_QWORD *)(v21 + 104) == 0
                   && *(_BYTE *)(v21 + 40) == 0
                   && *(_BYTE *)(v21 + 80) == 1;
                if ( v73 && *(_BYTE *)(v21 + 144) == 1 )
                {
                  v78 = *(_DWORD *)(v21 + 48);
                  v22 = sub_2EBEE90(v17, v78);
                  v23 = v22;
                  if ( v22 )
                  {
                    if ( *(_WORD *)(v22 + 68) == 322 && (*(_DWORD *)(v22 + 40) & 0xFFFFFFu) > 2 )
                    {
                      v81 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 128LL))(*(_QWORD *)(a2 + 16));
                      v79 = sub_2EC06C0(
                              v17,
                              *(_QWORD *)(*(_QWORD *)(v17 + 56) + 16LL * (v78 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                              byte_3F871B3,
                              0,
                              v24,
                              v25);
                      v26 = sub_2EBEE90(v17, *(_DWORD *)(*(_QWORD *)(v23 + 32) + 48LL));
                      v27 = v26;
                      if ( v26 )
                      {
                        if ( *(_WORD *)(v26 + 68) == 3736 )
                        {
                          v28 = *(unsigned __int8 **)(v26 + 56);
                          v29 = *(_QWORD *)(v81 + 8);
                          v87[0] = v28;
                          v75 = v29 - 64320;
                          if ( v28 )
                            sub_B96E90((__int64)v87, (__int64)v28, 1);
                          v91.m128i_i64[0] = (__int64)v87[0];
                          if ( v87[0] )
                          {
                            sub_B976B0((__int64)v87, v87[0], (__int64)&v91);
                            v87[0] = 0;
                          }
                          v91.m128i_i64[1] = 0;
                          v92 = 0;
                          v30 = sub_36F43D0((_QWORD *)a2, (__int64)&v91, v75, v79);
                          v76 = v31;
                          sub_2E8EAD0(v31, (__int64)v30, (const __m128i *)(*(_QWORD *)(v23 + 32) + 40LL));
                          sub_9C6650(&v91);
                          sub_9C6650(v87);
                          v32 = *(__int64 **)(v27 + 24);
                          if ( v32 + 6 == (__int64 *)(v32[6] & 0xFFFFFFFFFFFFFFF8LL) )
                            v33 = (__int64 *)v32[7];
                          else
                            v33 = *(__int64 **)(v27 + 8);
                          sub_2E31040(v32 + 5, v76);
                          v34 = *(_QWORD *)v76;
                          v35 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v76 + 8) = v33;
                          *(_QWORD *)v76 = v35 | v34 & 7;
                          *(_QWORD *)(v35 + 8) = v76;
                          *v33 = v76 | *v33 & 7;
                          v36 = *(unsigned __int8 **)(v20 + 56);
                          v37 = *(_DWORD *)(*(_QWORD *)(v20 + 32) + 8LL);
                          v38 = *(_QWORD *)(v81 + 8);
                          v86 = v36;
                          v82 = v38 - 141840;
                          if ( v36 )
                            sub_B96E90((__int64)&v86, (__int64)v36, 1);
                          v87[0] = v86;
                          if ( v86 )
                          {
                            sub_B976B0((__int64)&v86, v86, (__int64)v87);
                            v86 = 0;
                          }
                          v87[1] = 0;
                          v87[2] = 0;
                          v39 = sub_36F43D0((_QWORD *)a2, (__int64)v87, v82, v37);
                          v41 = v40;
                          v83 = (__int64)v39;
                          sub_2E8EAD0(v40, (__int64)v39, (const __m128i *)(*(_QWORD *)(v23 + 32) + 80LL));
                          v91.m128i_i64[0] = 0;
                          v91.m128i_i32[2] = v79;
                          v92 = 0;
                          v93 = 0;
                          v94 = 0;
                          sub_2E8EAD0(v41, v83, &v91);
                          v91.m128i_i64[0] = 1;
                          v92 = 0;
                          v93 = 1;
                          sub_2E8EAD0(v41, v83, &v91);
                          sub_9C6650(v87);
                          sub_9C6650(&v86);
                          sub_2E31040(v72 + 5, v41);
                          v42 = *(_QWORD *)v5;
                          v43 = *(_QWORD *)v41;
                          *(_QWORD *)(v41 + 8) = v5;
                          v42 &= 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)v41 = v42 | v43 & 7;
                          *(_QWORD *)(v42 + 8) = v41;
                          *(_QWORD *)v5 = v41 | *(_QWORD *)v5 & 7LL;
                          sub_2E88E20(v20);
                          v6 = v73;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v5 = (__int64)v85;
    }
    while ( v85 != v72 + 6 );
    v3 = v6;
LABEL_15:
    v72 = (_QWORD *)v72[1];
  }
  while ( v70 != v72 );
LABEL_16:
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(__int64 (**)(void))(*(_QWORD *)v10 + 200LL);
  if ( (char *)v11 == (char *)sub_3020000 )
    v12 = v10 + 456;
  else
    v12 = v11();
  v13 = *(_QWORD *)(a2 + 32);
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 680LL))(v12);
  if ( v14 < 0 )
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 56) + 16LL * (v14 & 0x7FFFFFFF) + 8);
  else
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 304) + 8LL * (unsigned int)v14);
  if ( v15 )
  {
    if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
      return v3;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 32);
      if ( !v15 )
        break;
      if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
        return v3;
    }
  }
  v68 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 680LL))(v12, a2);
  v69 = sub_2EBEE90(v13, v68);
  if ( v69 )
    sub_2E88E20(v69);
  return v3;
}
