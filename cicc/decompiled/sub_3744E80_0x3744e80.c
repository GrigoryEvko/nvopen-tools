// Function: sub_3744E80
// Address: 0x3744e80
//
__int64 __fastcall sub_3744E80(__int64 a1, __int64 a2)
{
  unsigned __int16 v3; // ax
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r13
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r10
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 i; // rbx
  int v17; // eax
  __int64 v18; // r14
  const __m128i *v19; // r13
  __int64 (__fastcall *v20)(__int64, __int64, __int64, unsigned __int64); // r15
  __m128i v21; // xmm0
  __int64 v23; // rdx
  int v24; // r13d
  __int64 v25; // rax
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 (__fastcall *v29)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  unsigned __int16 v32; // dx
  __int64 v33; // rax
  char v34; // cl
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // edx
  int v39; // r11d
  unsigned __int64 v40; // rdx
  __int128 v41; // [rsp-10h] [rbp-160h]
  __int64 v42; // [rsp+8h] [rbp-148h]
  __int64 v43; // [rsp+10h] [rbp-140h]
  __int64 v44; // [rsp+18h] [rbp-138h]
  __int64 v45; // [rsp+20h] [rbp-130h]
  unsigned __int64 v46; // [rsp+20h] [rbp-130h]
  __int64 v48; // [rsp+48h] [rbp-108h]
  unsigned int v49; // [rsp+54h] [rbp-FCh]
  __int64 v50; // [rsp+58h] [rbp-F8h]
  unsigned __int16 v51; // [rsp+66h] [rbp-EAh] BYREF
  unsigned int v52; // [rsp+68h] [rbp-E8h] BYREF
  unsigned int v53; // [rsp+6Ch] [rbp-E4h]
  __m128i v54; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v55; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v56; // [rsp+90h] [rbp-C0h]
  __int64 v57; // [rsp+98h] [rbp-B8h]
  __int64 v58; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-A8h]
  __int64 v60; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-98h]
  unsigned __int64 v62; // [rsp+C0h] [rbp-90h]
  unsigned __int64 v63[2]; // [rsp+D0h] [rbp-80h] BYREF
  _BYTE v64[112]; // [rsp+E0h] [rbp-70h] BYREF

  v3 = sub_2D5BAE0(*(_QWORD *)(a1 + 128), *(_QWORD *)(a1 + 112), *(__int64 **)(a2 + 8), 1);
  if ( !v3 || !*(_QWORD *)(*(_QWORD *)(a1 + 128) + 8LL * v3 + 112) && v3 != 2 )
    return 0;
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(unsigned int *)(v4 + 144);
  v7 = *(_QWORD *)(v4 + 128);
  v8 = *(_QWORD *)(v5 + 8);
  if ( !(_DWORD)v6 )
  {
LABEL_37:
    if ( *(_BYTE *)v5 > 0x1Cu )
    {
      v49 = sub_374D810(v4, v5);
      goto LABEL_8;
    }
    return 0;
  }
  v9 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v5 != *v10 )
  {
    v38 = 1;
    while ( v11 != -4096 )
    {
      v39 = v38 + 1;
      v9 = (v6 - 1) & (v38 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v5 == *v10 )
        goto LABEL_6;
      v38 = v39;
    }
    goto LABEL_37;
  }
LABEL_6:
  if ( v10 == (__int64 *)(v7 + 16 * v6) )
    goto LABEL_37;
  v49 = *((_DWORD *)v10 + 2);
LABEL_8:
  v12 = sub_34B8B90(v8, *(_DWORD **)(a2 + 72), (_DWORD *)(*(_QWORD *)(a2 + 72) + 4LL * *(unsigned int *)(a2 + 80)), 0);
  v13 = *(_QWORD *)(a1 + 112);
  LOBYTE(v61) = 0;
  v14 = v12;
  v15 = *(_QWORD *)(a1 + 128);
  v63[0] = (unsigned __int64)v64;
  *((_QWORD *)&v41 + 1) = v61;
  v60 = 0;
  *(_QWORD *)&v41 = 0;
  v63[1] = 0x400000000LL;
  sub_34B8C80(v15, v13, v8, (__int64)v63, 0, 0, v41);
  if ( (_DWORD)v14 )
  {
    v48 = 16 * v14;
    for ( i = 0; v48 != i; i += 16 )
    {
      v18 = *(_QWORD *)(a1 + 128);
      BYTE2(v53) = 0;
      v19 = (const __m128i *)(i + v63[0]);
      v20 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v18 + 736LL);
      v50 = sub_B2BE50(**(_QWORD **)(a1 + 40));
      if ( v20 == sub_2FEA1A0 )
      {
        v21 = _mm_loadu_si128(v19);
        v54 = v21;
        if ( v21.m128i_i16[0] )
        {
          v17 = *(unsigned __int16 *)(v18 + 2LL * v21.m128i_u16[0] + 2304);
        }
        else
        {
          if ( !sub_30070B0((__int64)&v54) )
          {
            if ( !sub_3007070((__int64)&v54) )
              goto LABEL_48;
            v56 = sub_3007260((__int64)&v54);
            v57 = v23;
            v60 = v56;
            LOBYTE(v61) = v23;
            v24 = sub_CA1930(&v60);
            v25 = v54.m128i_u16[0];
            v55 = v54;
            if ( v54.m128i_i16[0] )
              goto LABEL_29;
            v44 = v54.m128i_i64[1];
            v45 = v54.m128i_i64[0];
            if ( sub_30070B0((__int64)&v55) )
            {
              LOWORD(v60) = 0;
              LOWORD(v52) = 0;
              v61 = 0;
              sub_2FE8D10(
                v18,
                v50,
                v55.m128i_u32[0],
                v55.m128i_u64[1],
                &v60,
                (unsigned int *)&v58,
                (unsigned __int16 *)&v52);
              v32 = v52;
            }
            else
            {
              if ( !sub_3007070((__int64)&v55) )
                goto LABEL_48;
              v26 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v18 + 592LL);
              if ( v26 == sub_2D56A50 )
              {
                sub_2FE6CC0((__int64)&v60, v18, v50, v45, v44);
                v27 = v43;
                LOWORD(v27) = v61;
                v28 = v62;
                v43 = v27;
              }
              else
              {
                v43 = v26(v18, v50, v55.m128i_u32[0], v55.m128i_i64[1]);
                v28 = v37;
              }
              v59 = v28;
              v25 = (unsigned __int16)v43;
              v58 = v43;
              if ( !(_WORD)v43 )
              {
                v46 = v28;
                if ( sub_30070B0((__int64)&v58) )
                {
                  LOWORD(v60) = 0;
                  v51 = 0;
                  v61 = 0;
                  sub_2FE8D10(v18, v50, (unsigned int)v58, v46, &v60, &v52, &v51);
                  v32 = v51;
                }
                else
                {
                  if ( !sub_3007070((__int64)&v58) )
LABEL_48:
                    BUG();
                  v29 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v18 + 592LL);
                  if ( v29 == sub_2D56A50 )
                  {
                    sub_2FE6CC0((__int64)&v60, v18, v50, v58, v59);
                    v30 = v42;
                    LOWORD(v30) = v61;
                    v31 = v62;
                    v42 = v30;
                  }
                  else
                  {
                    v42 = v29(v18, v50, v58, v46);
                    v31 = v40;
                  }
                  v32 = sub_2FE98B0(v18, v50, (unsigned int)v42, v31);
                }
                goto LABEL_30;
              }
LABEL_29:
              v32 = *(_WORD *)(v18 + 2 * v25 + 2852);
            }
LABEL_30:
            if ( v32 <= 1u || (unsigned __int16)(v32 - 504) <= 7u )
              BUG();
            v33 = 16LL * (v32 - 1);
            v34 = byte_444C4A0[v33 + 8];
            v35 = *(_QWORD *)&byte_444C4A0[v33];
            LOBYTE(v61) = v34;
            v60 = v35;
            v36 = sub_CA1930(&v60);
            v17 = (v24 + v36 - 1) / v36;
            goto LABEL_11;
          }
          LOWORD(v60) = 0;
          v61 = 0;
          v55.m128i_i16[0] = 0;
          v17 = sub_2FE8D10(
                  v18,
                  v50,
                  v54.m128i_u32[0],
                  v54.m128i_u64[1],
                  &v60,
                  (unsigned int *)&v58,
                  (unsigned __int16 *)&v55);
        }
      }
      else
      {
        v17 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, _QWORD))v20)(
                v18,
                v50,
                v19->m128i_u32[0],
                v19->m128i_i64[1],
                v53);
      }
LABEL_11:
      v49 += v17;
    }
  }
  sub_3742B00(a1, (_BYTE *)a2, v49, 1);
  if ( (_BYTE *)v63[0] != v64 )
    _libc_free(v63[0]);
  return 1;
}
