// Function: sub_15450E0
// Address: 0x15450e0
//
void __fastcall sub_15450E0(__int64 a1, int a2, __int64 a3)
{
  int v3; // r12d
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rcx
  _BYTE *v12; // rdi
  __int64 v13; // r13
  __int64 *v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 *v17; // r15
  __int64 v18; // r15
  _BYTE *v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // r14
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // r13
  __int64 v30; // r12
  _BYTE *v31; // rbx
  __int64 v32; // r14
  __int64 v33; // r15
  _QWORD *v34; // rax
  int v35; // r10d
  __int64 *v36; // r11
  int v37; // ecx
  int v38; // ecx
  int v39; // r9d
  int v40; // r9d
  __int64 v41; // r10
  __int64 v42; // rdx
  __int64 v43; // r8
  int v44; // edi
  __int64 *v45; // rsi
  int v46; // edi
  int v47; // edi
  __int64 v48; // r8
  int v49; // edx
  __int64 *v50; // r9
  __int64 v51; // r15
  __int64 v52; // rsi
  int v53; // [rsp+Ch] [rbp-384h]
  __int64 v54; // [rsp+38h] [rbp-358h] BYREF
  _BYTE *v55; // [rsp+40h] [rbp-350h] BYREF
  __int64 v56; // [rsp+48h] [rbp-348h]
  _BYTE v57[256]; // [rsp+50h] [rbp-340h] BYREF
  _BYTE *v58; // [rsp+150h] [rbp-240h] BYREF
  __int64 v59; // [rsp+158h] [rbp-238h]
  _BYTE v60[560]; // [rsp+160h] [rbp-230h] BYREF

  v3 = a2;
  v4 = a1;
  v55 = v57;
  v56 = 0x2000000000LL;
  v58 = v60;
  v59 = 0x2000000000LL;
  v5 = sub_1544DA0(a1, a2, a3);
  if ( v5 )
  {
    v6 = v5;
    v7 = v5 - 8LL * *(unsigned int *)(v5 + 8);
    v8 = (unsigned int)v59;
    if ( (unsigned int)v59 >= HIDWORD(v59) )
    {
      sub_16CD150(&v58, v60, 0, 16);
      v8 = (unsigned int)v59;
    }
    v9 = (__int64 *)&v58[16 * v8];
    *v9 = v6;
    v9[1] = v7;
    v10 = (unsigned int)(v59 + 1);
    LODWORD(v59) = v59 + 1;
  }
  else
  {
    v10 = (unsigned int)v59;
  }
  v11 = v58;
  v12 = v58;
  if ( (_DWORD)v10 )
  {
    v13 = *(_QWORD *)&v58[16 * v10 - 16];
    while ( 1 )
    {
      v14 = *(__int64 **)&v11[16 * v10 - 8];
      v15 = (v13 - (__int64)v14) >> 5;
      v16 = (v13 - (__int64)v14) >> 3;
      if ( v15 <= 0 )
        goto LABEL_31;
      v17 = &v14[4 * v15];
      do
      {
        if ( sub_1544DA0(v4, v3, *v14) )
          goto LABEL_14;
        if ( sub_1544DA0(v4, v3, v14[1]) )
        {
          ++v14;
          goto LABEL_14;
        }
        if ( sub_1544DA0(v4, v3, v14[2]) )
        {
          v14 += 2;
          goto LABEL_14;
        }
        if ( sub_1544DA0(v4, v3, v14[3]) )
        {
          v14 += 3;
          goto LABEL_14;
        }
        v14 += 4;
      }
      while ( v17 != v14 );
      v16 = (v13 - (__int64)v14) >> 3;
LABEL_31:
      switch ( v16 )
      {
        case 2LL:
          goto LABEL_47;
        case 3LL:
          if ( sub_1544DA0(v4, v3, *v14) )
            goto LABEL_14;
          ++v14;
LABEL_47:
          if ( !sub_1544DA0(v4, v3, *v14) )
          {
            ++v14;
            goto LABEL_49;
          }
LABEL_14:
          if ( v14 == (__int64 *)v13 )
            goto LABEL_34;
          v18 = *v14;
          v19 = v58;
          *(_QWORD *)&v58[16 * (unsigned int)v59 - 8] = v14 + 1;
          if ( *(_BYTE *)(v18 + 1) == 1 && *(_BYTE *)(v13 + 1) != 1 )
          {
            v20 = (unsigned int)v56;
            if ( (unsigned int)v56 >= HIDWORD(v56) )
            {
              sub_16CD150(&v55, v57, 0, 8);
              v20 = (unsigned int)v56;
            }
            *(_QWORD *)&v55[8 * v20] = v18;
            LODWORD(v20) = v59;
            LODWORD(v56) = v56 + 1;
            goto LABEL_20;
          }
          v21 = v18 - 8LL * *(unsigned int *)(v18 + 8);
          v22 = (unsigned int)v59;
          if ( (unsigned int)v59 >= HIDWORD(v59) )
          {
            sub_16CD150(&v58, v60, 0, 16);
            v19 = v58;
            v22 = (unsigned int)v59;
          }
          v20 = (unsigned __int64)&v19[16 * v22];
          *(_QWORD *)v20 = v18;
          v11 = v58;
          *(_QWORD *)(v20 + 8) = v21;
          v12 = v11;
          LODWORD(v20) = v59 + 1;
          LODWORD(v59) = v20;
          if ( !(_DWORD)v20 )
            goto LABEL_25;
LABEL_21:
          v10 = (unsigned int)v20;
          v13 = *(_QWORD *)&v11[16 * (unsigned int)v20 - 16];
          break;
        case 1LL:
LABEL_49:
          if ( sub_1544DA0(v4, v3, *v14) )
            goto LABEL_14;
LABEL_34:
          LODWORD(v59) = v59 - 1;
          v54 = v13;
          sub_153F6C0(v4 + 208, &v54);
          v23 = *(_DWORD *)(v4 + 280);
          v24 = (__int64)(*(_QWORD *)(v4 + 216) - *(_QWORD *)(v4 + 208)) >> 3;
          if ( !v23 )
          {
            ++*(_QWORD *)(v4 + 256);
            goto LABEL_65;
          }
          v25 = *(_QWORD *)(v4 + 264);
          v26 = (v23 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( *v27 != v13 )
          {
            v35 = 1;
            v36 = 0;
            while ( v28 != -4 )
            {
              if ( !v36 && v28 == -8 )
                v36 = v27;
              v26 = (v23 - 1) & (v35 + v26);
              v27 = (__int64 *)(v25 + 16LL * v26);
              v28 = *v27;
              if ( *v27 == v13 )
                goto LABEL_36;
              ++v35;
            }
            v37 = *(_DWORD *)(v4 + 272);
            if ( v36 )
              v27 = v36;
            ++*(_QWORD *)(v4 + 256);
            v38 = v37 + 1;
            if ( 4 * v38 >= 3 * v23 )
            {
LABEL_65:
              sub_1542590(v4 + 256, 2 * v23);
              v39 = *(_DWORD *)(v4 + 280);
              if ( !v39 )
                goto LABEL_93;
              v40 = v39 - 1;
              v41 = *(_QWORD *)(v4 + 264);
              v38 = *(_DWORD *)(v4 + 272) + 1;
              LODWORD(v42) = v40 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
              v27 = (__int64 *)(v41 + 16LL * (unsigned int)v42);
              v43 = *v27;
              if ( *v27 != v13 )
              {
                v44 = 1;
                v45 = 0;
                while ( v43 != -4 )
                {
                  if ( v43 == -8 && !v45 )
                    v45 = v27;
                  v42 = v40 & (unsigned int)(v42 + v44);
                  v27 = (__int64 *)(v41 + 16 * v42);
                  v43 = *v27;
                  if ( *v27 == v13 )
                    goto LABEL_61;
                  ++v44;
                }
                if ( v45 )
                  v27 = v45;
              }
            }
            else if ( v23 - *(_DWORD *)(v4 + 276) - v38 <= v23 >> 3 )
            {
              sub_1542590(v4 + 256, v23);
              v46 = *(_DWORD *)(v4 + 280);
              if ( !v46 )
              {
LABEL_93:
                ++*(_DWORD *)(v4 + 272);
                BUG();
              }
              v47 = v46 - 1;
              v48 = *(_QWORD *)(v4 + 264);
              v49 = 1;
              v50 = 0;
              LODWORD(v51) = v47 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
              v38 = *(_DWORD *)(v4 + 272) + 1;
              v27 = (__int64 *)(v48 + 16LL * (unsigned int)v51);
              v52 = *v27;
              if ( *v27 != v13 )
              {
                while ( v52 != -4 )
                {
                  if ( !v50 && v52 == -8 )
                    v50 = v27;
                  v51 = v47 & (unsigned int)(v51 + v49);
                  v27 = (__int64 *)(v48 + 16 * v51);
                  v52 = *v27;
                  if ( *v27 == v13 )
                    goto LABEL_61;
                  ++v49;
                }
                if ( v50 )
                  v27 = v50;
              }
            }
LABEL_61:
            *(_DWORD *)(v4 + 272) = v38;
            if ( *v27 != -4 )
              --*(_DWORD *)(v4 + 276);
            *v27 = v13;
            v27[1] = 0;
          }
LABEL_36:
          *((_DWORD *)v27 + 3) = v24;
          v20 = (unsigned int)v59;
          if ( !(_DWORD)v59
            || (v10 = (unsigned int)v59,
                v11 = v58,
                v13 = *(_QWORD *)&v58[16 * (unsigned int)v59 - 16],
                *(_BYTE *)(v13 + 1) == 1) )
          {
            v29 = (unsigned __int64)v55;
            if ( v55 != &v55[8 * (unsigned int)v56] )
            {
              v53 = v3;
              v30 = v4;
              v31 = &v55[8 * (unsigned int)v56];
              do
              {
                v32 = *(_QWORD *)v29;
                v33 = *(_QWORD *)v29 - 8LL * *(unsigned int *)(*(_QWORD *)v29 + 8LL);
                if ( HIDWORD(v59) <= (unsigned int)v20 )
                {
                  sub_16CD150(&v58, v60, 0, 16);
                  v20 = (unsigned int)v59;
                }
                v34 = &v58[16 * v20];
                v29 += 8LL;
                *v34 = v32;
                v34[1] = v33;
                v20 = (unsigned int)(v59 + 1);
                LODWORD(v59) = v59 + 1;
              }
              while ( v31 != (_BYTE *)v29 );
              v4 = v30;
              v3 = v53;
            }
            LODWORD(v56) = 0;
LABEL_20:
            v11 = v58;
            v12 = v58;
            if ( !(_DWORD)v20 )
              goto LABEL_25;
            goto LABEL_21;
          }
          break;
        default:
          goto LABEL_34;
      }
    }
  }
LABEL_25:
  if ( v12 != v60 )
    _libc_free((unsigned __int64)v12);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
}
