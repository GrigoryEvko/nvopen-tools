// Function: sub_10E1270
// Address: 0x10e1270
//
__int64 __fastcall sub_10E1270(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned __int8 v5; // r14
  unsigned __int16 v6; // ax
  unsigned __int8 v7; // r14
  unsigned __int16 v8; // ax
  __int64 v9; // r14
  char v10; // al
  unsigned __int8 v11; // r8
  unsigned __int8 v12; // r10
  __int64 v13; // r14
  char *v14; // rdx
  unsigned __int8 v15; // al
  unsigned int v16; // r14d
  unsigned __int64 v17; // r14
  __int64 v18; // rax
  int v19; // eax
  _QWORD *v20; // rax
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r12
  __int64 v36; // rax
  char v37; // al
  _QWORD *v38; // rax
  __int64 v39; // r9
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // r12
  __int64 v44; // rbx
  __int64 v45; // rdx
  unsigned int v46; // esi
  __int64 v47; // rax
  int v48; // edx
  __int64 v49; // rax
  unsigned int v50; // r12d
  bool v51; // dl
  __int64 v52; // rax
  unsigned int v53; // r12d
  bool v54; // dl
  int v55; // eax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 *v58; // rax
  __int64 *v59; // rax
  __int64 v60; // r12
  __int64 *v61; // rax
  __int64 *v63; // rax
  __int64 *v64; // rax
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rcx
  __int64 v68; // rax
  char *v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdi
  unsigned int v73; // r15d
  __int16 v74; // ax
  __int16 v75; // ax
  __int64 v76; // [rsp+0h] [rbp-F0h]
  char v77; // [rsp+0h] [rbp-F0h]
  __int64 v78; // [rsp+10h] [rbp-E0h]
  __int64 v79; // [rsp+10h] [rbp-E0h]
  char v80; // [rsp+18h] [rbp-D8h]
  __int64 v81; // [rsp+18h] [rbp-D8h]
  __int16 v82; // [rsp+20h] [rbp-D0h]
  __int16 v83; // [rsp+24h] [rbp-CCh]
  unsigned __int8 v84; // [rsp+28h] [rbp-C8h]
  __int64 v85; // [rsp+28h] [rbp-C8h]
  __int64 v87; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v88; // [rsp+30h] [rbp-C0h]
  unsigned __int8 v89; // [rsp+30h] [rbp-C0h]
  unsigned __int8 v90; // [rsp+30h] [rbp-C0h]
  unsigned __int8 v91; // [rsp+38h] [rbp-B8h]
  __int64 v92; // [rsp+38h] [rbp-B8h]
  unsigned __int8 v93; // [rsp+38h] [rbp-B8h]
  __m128i v94[2]; // [rsp+40h] [rbp-B0h] BYREF
  _BYTE v95[32]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v96; // [rsp+80h] [rbp-70h]
  __m128i v97; // [rsp+90h] [rbp-60h] BYREF
  __int64 v98; // [rsp+A0h] [rbp-50h]
  __int64 v99; // [rsp+A8h] [rbp-48h]
  __int64 v100; // [rsp+B0h] [rbp-40h]
  __int64 v101; // [rsp+B8h] [rbp-38h]

  v3 = a2;
  v4 = (__int64 *)(a2 + 72);
  v5 = sub_F518D0(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0, a1[11], a2, a1[8], a1[10]);
  v6 = sub_A74840((_QWORD *)(a2 + 72), 0);
  if ( !HIBYTE(v6) || (unsigned __int8)v6 < v5 )
  {
    v63 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B980(v4, v63, 1, 86);
    v64 = (__int64 *)sub_BD5C60(a2);
    v97.m128i_i32[0] = 0;
    v60 = sub_A77A40(v64, v5);
    goto LABEL_60;
  }
  v91 = v6;
  v7 = sub_F518D0(
         *(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
         0,
         a1[11],
         a2,
         a1[8],
         a1[10]);
  v8 = sub_A74840(v4, 1);
  if ( !HIBYTE(v8) || (v88 = v8, (unsigned __int8)v8 < v7) )
  {
    v58 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B980(v4, v58, 2, 86);
    v59 = (__int64 *)sub_BD5C60(a2);
    v97.m128i_i32[0] = 1;
    v60 = sub_A77A40(v59, v7);
LABEL_60:
    v61 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B660(v4, v61, &v97, 1, v60);
    return a2;
  }
  v9 = a1[7];
  v97.m128i_i64[0] = (__int64)sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 1);
  v97.m128i_i64[1] = -1;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v10 = sub_CF5020(v9, (__int64)&v97, 0);
  v11 = v88;
  v12 = v91;
  if ( (v10 & 2) == 0 )
  {
    v65 = 2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    goto LABEL_63;
  }
  v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v14 = *(char **)(a2 + 32 * (1 - v13));
  v15 = *v14;
  if ( (unsigned __int8)*v14 <= 0x1Cu )
    goto LABEL_9;
  while ( v15 == 63 )
  {
    v68 = *((_QWORD *)v14 + 2);
    if ( v68 && !*(_QWORD *)(v68 + 8) )
    {
      v69 = (v14[7] & 0x40) != 0 ? (char *)*((_QWORD *)v14 - 1) : &v14[-32 * (*((_DWORD *)v14 + 1) & 0x7FFFFFF)];
      v14 = *(char **)v69;
      v15 = *v14;
      if ( (unsigned __int8)*v14 > 0x1Cu )
        continue;
    }
    goto LABEL_9;
  }
  if ( v15 != 60 )
    goto LABEL_9;
  v70 = *((_QWORD *)v14 + 2);
  if ( !v70 || *(_QWORD *)(v70 + 8) )
    goto LABEL_9;
  v71 = *(_QWORD *)(a2 - 32);
  if ( !v71 || *(_BYTE *)v71 || *(_QWORD *)(v71 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_100;
  if ( (unsigned int)(*(_DWORD *)(v71 + 36) - 238) <= 7 && ((1LL << (*(_BYTE *)(v71 + 36) + 18)) & 0xAD) != 0 )
  {
    v72 = *(_QWORD *)(a2 + 32 * (3 - v13));
    v73 = *(_DWORD *)(v72 + 32);
    if ( v73 > 0x40 )
    {
      v90 = v91;
      v93 = v11;
      if ( v73 != (unsigned int)sub_C444A0(v72 + 24) )
      {
        v11 = v93;
        v12 = v90;
        goto LABEL_9;
      }
      goto LABEL_92;
    }
    if ( *(_QWORD *)(v72 + 24) )
    {
LABEL_9:
      v92 = *(_QWORD *)(a2 + 32 * (2 - v13));
      if ( *(_BYTE *)v92 == 17 )
      {
        v16 = *(_DWORD *)(v92 + 32);
        if ( v16 <= 0x40 )
        {
          v17 = *(_QWORD *)(v92 + 24);
        }
        else
        {
          v84 = v12;
          v89 = v11;
          if ( v16 - (unsigned int)sub_C444A0(v92 + 24) > 0x40 )
            return 0;
          v11 = v89;
          v12 = v84;
          v17 = **(_QWORD **)(v92 + 24);
        }
        if ( v17 <= 8 && (v17 & (v17 - 1)) == 0 )
        {
          v18 = *(_QWORD *)(a2 - 32);
          if ( !v18 || *(_BYTE *)v18 || *(_QWORD *)(v18 + 24) != *(_QWORD *)(a2 + 80) )
            goto LABEL_100;
          v19 = *(_DWORD *)(v18 + 36);
          v83 = v12;
          if ( v19 != 239 && v19 != 242 )
          {
            v82 = v11;
            goto LABEL_21;
          }
          if ( 1LL << v12 >= v17 )
          {
            v82 = v11;
            if ( 1LL << v11 >= v17 )
            {
LABEL_21:
              v20 = (_QWORD *)sub_BD5C60(a2);
              v85 = sub_BCCE00(v20, 8 * (int)v17);
              sub_B91FC0(v97.m128i_i64, a2);
              sub_E00CC0(v94, &v97, v17);
              v21 = (__int64 *)a1[4];
              v22 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
              v23 = v21[6];
              v96 = 257;
              v76 = *(_QWORD *)(a2 + 32 * (1 - v22));
              v78 = *(_QWORD *)(a2 - 32 * v22);
              v24 = sub_AA4E30(v23);
              v80 = sub_AE5020(v24, v85);
              LOWORD(v100) = 257;
              v25 = sub_BD2C40(80, unk_3F10A14);
              v26 = (__int64)v25;
              if ( v25 )
                sub_B4D190((__int64)v25, v85, v76, (__int64)&v97, 0, v80, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v21[11] + 16LL))(
                v21[11],
                v26,
                v95,
                v21[7],
                v21[8]);
              v27 = *v21;
              v28 = 16LL * *((unsigned int *)v21 + 2);
              if ( v27 != v27 + v28 )
              {
                v29 = v27 + v28;
                v30 = v27;
                do
                {
                  v31 = *(_QWORD *)(v30 + 8);
                  v32 = *(_DWORD *)v30;
                  v30 += 16;
                  sub_B99FD0(v26, v32, v31);
                }
                while ( v29 != v30 );
                v3 = a2;
              }
              *(_WORD *)(v26 + 2) = (2 * v82) | *(_WORD *)(v26 + 2) & 0xFF81;
              sub_B9A100(v26, v94[0].m128i_i64);
              if ( (*(_BYTE *)(v3 + 7) & 0x20) != 0 )
              {
                v33 = sub_B91C10(v3, 10);
                v87 = v33;
                if ( v33 )
                  sub_B99FD0(v26, 0xAu, v33);
                if ( (*(_BYTE *)(v3 + 7) & 0x20) != 0 )
                {
                  v34 = sub_B91C10(v3, 25);
                  v81 = v34;
                  if ( v34 )
                    sub_B99FD0(v26, 0x19u, v34);
                }
                else
                {
                  v81 = 0;
                }
              }
              else
              {
                v87 = 0;
                v81 = 0;
              }
              v35 = (__int64 *)a1[4];
              v36 = sub_AA4E30(v35[6]);
              v37 = sub_AE5020(v36, *(_QWORD *)(v26 + 8));
              LOWORD(v100) = 257;
              v77 = v37;
              v38 = sub_BD2C40(80, unk_3F10A10);
              v40 = (__int64)v38;
              if ( v38 )
                sub_B4D3C0((__int64)v38, v26, v78, 0, v77, v39, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v35[11] + 16LL))(
                v35[11],
                v40,
                &v97,
                v35[7],
                v35[8]);
              v41 = *v35;
              v42 = 16LL * *((unsigned int *)v35 + 2);
              if ( v41 != v41 + v42 )
              {
                v79 = v3;
                v43 = v41 + v42;
                v44 = v41;
                do
                {
                  v45 = *(_QWORD *)(v44 + 8);
                  v46 = *(_DWORD *)v44;
                  v44 += 16;
                  sub_B99FD0(v40, v46, v45);
                }
                while ( v43 != v44 );
                v3 = v79;
              }
              *(_WORD *)(v40 + 2) = (2 * v83) | *(_WORD *)(v40 + 2) & 0xFF81;
              sub_B9A100(v40, v94[0].m128i_i64);
              if ( v87 )
                sub_B99FD0(v40, 0xAu, v87);
              if ( v81 )
                sub_B99FD0(v40, 0x19u, v81);
              v97.m128i_i32[0] = 38;
              sub_B47C00(v40, v3, v97.m128i_i32, 1);
              v47 = *(_QWORD *)(v3 - 32);
              if ( !v47 )
                BUG();
              if ( !*(_BYTE *)v47 && *(_QWORD *)(v47 + 24) == *(_QWORD *)(v3 + 80) )
              {
                v48 = *(_DWORD *)(v47 + 36);
                if ( v48 != 238 && (unsigned int)(v48 - 240) > 1 )
                  goto LABEL_54;
                v49 = *(_QWORD *)(v3 + 32 * (3LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
                v50 = *(_DWORD *)(v49 + 32);
                v51 = v50 <= 0x40 ? *(_QWORD *)(v49 + 24) == 0 : v50 == (unsigned int)sub_C444A0(v49 + 24);
                *(_WORD *)(v26 + 2) = !v51 | *(_WORD *)(v26 + 2) & 0xFFFE;
                v52 = *(_QWORD *)(v3 + 32 * (3LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
                v53 = *(_DWORD *)(v52 + 32);
                v54 = v53 <= 0x40 ? *(_QWORD *)(v52 + 24) == 0 : v53 == (unsigned int)sub_C444A0(v52 + 24);
                *(_WORD *)(v40 + 2) = !v54 | *(_WORD *)(v40 + 2) & 0xFFFE;
                v47 = *(_QWORD *)(v3 - 32);
                if ( v47 )
                {
                  if ( !*(_BYTE *)v47 )
                  {
LABEL_54:
                    if ( *(_QWORD *)(v47 + 24) == *(_QWORD *)(v3 + 80) )
                    {
                      v55 = *(_DWORD *)(v47 + 36);
                      if ( v55 == 239 || v55 == 242 )
                      {
                        v74 = *(_WORD *)(v26 + 2) & 0xFC7F;
                        LOBYTE(v74) = v74 | 0x80;
                        *(_WORD *)(v26 + 2) = v74;
                        v75 = *(_WORD *)(v40 + 2) & 0xFC7F;
                        LOBYTE(v75) = v75 | 0x80;
                        *(_WORD *)(v40 + 2) = v75;
                      }
                      v56 = sub_AD6530(*(_QWORD *)(v92 + 8), v3);
                      v57 = v3 + 32 * (2LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
                      if ( !*(_QWORD *)v57 )
                        goto LABEL_66;
                      goto LABEL_64;
                    }
                  }
                }
              }
LABEL_100:
              BUG();
            }
          }
        }
      }
      return 0;
    }
  }
LABEL_92:
  v65 = 2 - v13;
LABEL_63:
  v56 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a2 + 32 * v65) + 8LL), (__int64)&v97);
  v57 = a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v57 )
  {
LABEL_64:
    v66 = *(_QWORD *)(v57 + 8);
    **(_QWORD **)(v57 + 16) = v66;
    if ( v66 )
      *(_QWORD *)(v66 + 16) = *(_QWORD *)(v57 + 16);
  }
LABEL_66:
  *(_QWORD *)v57 = v56;
  if ( v56 )
  {
    v67 = *(_QWORD *)(v56 + 16);
    *(_QWORD *)(v57 + 8) = v67;
    if ( v67 )
      *(_QWORD *)(v67 + 16) = v57 + 8;
    *(_QWORD *)(v57 + 16) = v56 + 16;
    *(_QWORD *)(v56 + 16) = v57;
  }
  return v3;
}
