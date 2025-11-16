// Function: sub_25883A0
// Address: 0x25883a0
//
__int64 __fastcall sub_25883A0(__int64 a1, __int64 a2)
{
  __m128i *v3; // r14
  _BYTE *v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // rbx
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // rdi
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int v21; // edx
  __int64 *v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned __int64 v25; // rdi
  unsigned __int8 *v26; // r14
  __int64 (__fastcall *v27)(__int64); // rax
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  unsigned int v30; // esi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 (__fastcall *v36)(__int64); // rax
  char v37; // r8
  char v38; // al
  int v39; // ebx
  unsigned __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  _QWORD *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  char v48; // r12
  unsigned int *v49; // rax
  const __m128i *v50; // rbx
  __int64 v51; // rax
  int v52; // r10d
  __int64 v53; // rdx
  unsigned int v54; // eax
  _QWORD *v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // [rsp-8h] [rbp-F8h]
  _QWORD *v58; // [rsp+8h] [rbp-E8h]
  _QWORD *v59; // [rsp+10h] [rbp-E0h]
  __int64 v60; // [rsp+18h] [rbp-D8h]
  __int64 v61; // [rsp+30h] [rbp-C0h]
  int v62; // [rsp+38h] [rbp-B8h]
  char v63; // [rsp+38h] [rbp-B8h]
  char v64; // [rsp+3Eh] [rbp-B2h]
  char v65; // [rsp+3Fh] [rbp-B1h]
  __int64 v66; // [rsp+40h] [rbp-B0h]
  unsigned int v67; // [rsp+48h] [rbp-A8h]
  __int64 v68; // [rsp+48h] [rbp-A8h]
  char v69; // [rsp+5Bh] [rbp-95h] BYREF
  unsigned int v70; // [rsp+5Ch] [rbp-94h] BYREF
  __int64 v71; // [rsp+60h] [rbp-90h] BYREF
  __int64 v72; // [rsp+68h] [rbp-88h]
  __m128i v73; // [rsp+70h] [rbp-80h]
  __m128i v74; // [rsp+80h] [rbp-70h] BYREF
  _QWORD v75[12]; // [rsp+90h] [rbp-60h] BYREF

  v3 = (__m128i *)(a1 + 72);
  v5 = (_BYTE *)sub_2509740((_QWORD *)(a1 + 72));
  if ( !v5
    || (v6 = (__int64)v5, *v5 != 85)
    || (v18 = *((_QWORD *)v5 - 4)) == 0
    || *(_BYTE *)v18
    || *(_QWORD *)(v18 + 24) != *(_QWORD *)(v6 + 80)
    || (*(_BYTE *)(v18 + 33) & 0x20) == 0
    || (unsigned int)(*(_DWORD *)(v18 + 36) - 238) > 7
    || ((1LL << (*(_BYTE *)(v18 + 36) + 18)) & 0xAD) == 0 )
  {
    v7 = sub_250C680(v3->m128i_i64);
    v8 = v7;
    if ( v7 )
    {
      sub_250D230((unsigned __int64 *)&v74, v7, 6, 0);
      v9 = v74.m128i_i64[0];
      v10 = sub_252A820(a2, v74.m128i_i64[0], v74.m128i_i64[1], a1, 0, 0, 1);
      v13 = v57;
      v14 = (__int64 *)v10;
      if ( v10 )
      {
        v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 48LL);
        v16 = v15 == sub_2534F10
            ? (__int64)(v14 + 11)
            : ((__int64 (__fastcall *)(__int64 *, __int64, __int64, __int64))v15)(v14, v9, v11, v12);
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v16 + 16LL))(
               v16,
               v9,
               v11,
               v12,
               v13) )
        {
          v26 = (unsigned __int8 *)sub_2509740(v3);
          v27 = *(__int64 (__fastcall **)(__int64))(*v14 + 48);
          if ( v27 == sub_2534F10 )
            v28 = (__int64)(v14 + 11);
          else
            v28 = v27((__int64)v14);
          if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v28 + 16LL))(v28) )
          {
            v64 = *(_BYTE *)(a1 + 393);
            if ( v64 )
            {
              v70 = 1;
              v29 = sub_250C680(v14 + 9);
              v65 = sub_B2D680(v29);
              v30 = 1;
              if ( !sub_2553520(a1 + 288) )
              {
                v34 = (__int64)(v14 + 36);
                v35 = a1 + 288;
                if ( sub_2553520((__int64)(v14 + 36)) )
                {
                  sub_256B180(v35, v34, v31, v32, v33);
                  v30 = 0;
                }
                else
                {
                  v30 = (unsigned __int8)sub_256B200(v35, v34, v31, v32, v33) ^ 1;
                }
              }
              sub_250C0C0((int *)&v70, v30);
              v36 = *(__int64 (__fastcall **)(__int64))(*v14 + 48);
              if ( v36 == sub_2534F10 )
                v66 = (__int64)(v14 + 11);
              else
                v66 = v36((__int64)v14);
              if ( *(_DWORD *)(v66 + 152) )
              {
                v45 = *(_QWORD **)(v66 + 144);
                v46 = 12LL * *(unsigned int *)(v66 + 160);
                v58 = &v45[v46];
                while ( v58 != v45 )
                {
                  v59 = v45;
                  if ( *v45 == 0x7FFFFFFFFFFFFFFFLL )
                  {
                    if ( v45[1] != 0x7FFFFFFFFFFFFFFFLL )
                      goto LABEL_45;
                  }
                  else if ( *v45 != 0x7FFFFFFFFFFFFFFELL || v45[1] != 0x7FFFFFFFFFFFFFFELL )
                  {
LABEL_45:
                    if ( v45 == v58 )
                      return v70;
                    v60 = a1;
                    v61 = a1 + 88;
LABEL_47:
                    if ( v59[11] )
                    {
                      v48 = 0;
                      v47 = v59[9];
                      v68 = (__int64)(v59 + 7);
                    }
                    else
                    {
                      v47 = v59[2];
                      v48 = v64;
                      v68 = v47 + 4LL * *((unsigned int *)v59 + 6);
                    }
                    while ( 2 )
                    {
                      if ( v47 == v68 )
                      {
LABEL_58:
                        v55 = v59 + 12;
                        if ( v59 + 12 == v58 )
                          return v70;
                        while ( 1 )
                        {
                          if ( *v55 == 0x7FFFFFFFFFFFFFFFLL )
                          {
                            if ( v55[1] != 0x7FFFFFFFFFFFFFFFLL )
                              goto LABEL_61;
                          }
                          else if ( *v55 != 0x7FFFFFFFFFFFFFFELL || v55[1] != 0x7FFFFFFFFFFFFFFELL )
                          {
LABEL_61:
                            v59 = v55;
                            if ( v58 == v55 )
                              return v70;
                            goto LABEL_47;
                          }
                          v55 += 12;
                          if ( v58 == v55 )
                            return v70;
                        }
                      }
LABEL_50:
                      v49 = (unsigned int *)(v47 + 32);
                      if ( v48 )
                        v49 = (unsigned int *)v47;
                      v50 = (const __m128i *)(*(_QWORD *)(v66 + 8) + 112LL * *v49);
                      if ( v65 )
                      {
                        if ( (v50[6].m128i_i32[0] & 4) != 0 )
                        {
                          v62 = v50[6].m128i_i32[0] & 4;
                          v74 = _mm_loadu_si128(v50 + 1);
                          v69 = 0;
                          v73 = v74;
                          v51 = sub_2527B10(a2, v74.m128i_i64[0], v74.m128i_i64[1], v26, v60, &v69);
                          v52 = v62;
                          v71 = v51;
                          v72 = v53;
                          goto LABEL_55;
                        }
                      }
                      else
                      {
                        v63 = v50[6].m128i_i32[0];
                        v74 = _mm_loadu_si128(v50 + 1);
                        v69 = 0;
                        v73 = v74;
                        v71 = sub_2527B10(a2, v74.m128i_i64[0], v74.m128i_i64[1], v26, v60, &v69);
                        v72 = v56;
                        v52 = v63 & 0xC;
LABEL_55:
                        v54 = sub_256C1D0(
                                v61,
                                a2,
                                (__int64)v50[2].m128i_i64,
                                (__int64)v26,
                                v71,
                                v72,
                                (((v50[6].m128i_i8[0] & 2) != 0) + 1) | (unsigned int)v52,
                                v50[6].m128i_i64[1],
                                v50->m128i_i64[1]);
                        sub_250C0C0((int *)&v70, v54);
                      }
                      if ( !v48 )
                      {
                        v47 = sub_220EF30(v47);
                        continue;
                      }
                      break;
                    }
                    v47 += 4;
                    if ( v47 == v68 )
                      goto LABEL_58;
                    goto LABEL_50;
                  }
                  v45 += 12;
                }
              }
              return v70;
            }
          }
          goto LABEL_10;
        }
      }
      if ( !sub_B2FC80(*(_QWORD *)(v8 + 24)) )
        goto LABEL_10;
    }
    if ( (unsigned __int8)sub_2588040(a2, a1, v3->m128i_i64, 1, (bool *)&v70, 0, 0) )
    {
      LOBYTE(v71) = 0;
      v37 = sub_252A810(a2, v3, a1, (bool *)&v71);
      result = 1;
      if ( !v37 )
      {
        v38 = sub_252A800(a2, v3, a1, (bool *)&v71);
        v73.m128i_i64[0] = 0;
        v73.m128i_i64[1] = 1;
        v39 = -(v38 == 0);
        v40 = sub_2509740(v3);
        v74.m128i_i64[0] = (__int64)v75;
        v41 = v40;
        v74.m128i_i64[1] = 0x300000000LL;
        sub_2555810((__int64)&v74, 0x7FFFFFFF, 0x7FFFFFFF, v42, v43, v44);
        result = sub_256C1D0(a1 + 88, a2, (__int64)&v74, v41, v73.m128i_i64[0], v73.m128i_i64[1], (v39 & 8u) + 6, 0, 0);
        v25 = v74.m128i_i64[0];
        if ( (_QWORD *)v74.m128i_i64[0] != v75 )
          goto LABEL_24;
      }
      return result;
    }
LABEL_10:
    *(_BYTE *)(a1 + 393) = *(_BYTE *)(a1 + 392);
    return 0;
  }
  v19 = 0x7FFFFFFF;
  v20 = *(_QWORD *)(v6 + 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v20 == 17 )
  {
    v21 = *(_DWORD *)(v20 + 32);
    v22 = *(__int64 **)(v20 + 24);
    if ( v21 > 0x40 )
    {
      v19 = *v22;
    }
    else
    {
      v19 = 0;
      if ( v21 )
        v19 = (__int64)((_QWORD)v22 << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
    }
  }
  v23 = sub_250CB50(v3->m128i_i64, 0);
  if ( v23 > 1 )
    goto LABEL_10;
  v73.m128i_i64[0] = 0;
  v73.m128i_i64[1] = 1;
  v74.m128i_i64[0] = (__int64)v75;
  v75[0] = 0;
  v75[1] = v19;
  v74.m128i_i64[1] = 0x300000001LL;
  v24 = sub_256C1D0(a1 + 88, a2, (__int64)&v74, v6, 0, 1, v23 == 0 ? 9 : 5, 0, 0);
  result = sub_250C0B0(1, v24);
  v25 = v74.m128i_i64[0];
  if ( (_QWORD *)v74.m128i_i64[0] != v75 )
  {
LABEL_24:
    v67 = result;
    _libc_free(v25);
    return v67;
  }
  return result;
}
