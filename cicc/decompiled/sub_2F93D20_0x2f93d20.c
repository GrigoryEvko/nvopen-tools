// Function: sub_2F93D20
// Address: 0x2f93d20
//
char __fastcall sub_2F93D20(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  __int16 v5; // dx
  unsigned __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  unsigned int *v11; // rbx
  unsigned int *i; // r13
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  __int16 *v15; // r14
  unsigned int v16; // r12d
  _OWORD *v17; // rcx
  int v18; // edx
  unsigned int v19; // edi
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rbx
  _BYTE *v25; // r13
  _BYTE *v26; // r15
  __int64 v27; // rbx
  char v28; // al
  unsigned int v29; // eax
  __int64 v30; // rcx
  unsigned __int64 v31; // r8
  __int64 v32; // r9
  unsigned int v33; // r15d
  _BYTE *v34; // rbx
  _BYTE *v35; // r15
  int v36; // eax
  int v37; // eax
  char v38; // al
  bool v39; // cl
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // r15
  int v44; // ebx
  __int64 v45; // r14
  unsigned __int16 *v46; // rdi
  unsigned __int16 *v47; // rsi
  bool v49; // [rsp+Fh] [rbp-71h]
  __int64 *v50; // [rsp+18h] [rbp-68h]
  __int64 *v51; // [rsp+20h] [rbp-60h]
  unsigned __int16 *v52; // [rsp+20h] [rbp-60h]
  __int64 v53; // [rsp+28h] [rbp-58h]
  __int64 v54; // [rsp+30h] [rbp-50h]
  _OWORD *v55; // [rsp+38h] [rbp-48h]
  unsigned int v56; // [rsp+38h] [rbp-48h]
  __int64 v57; // [rsp+40h] [rbp-40h] BYREF
  int v58; // [rsp+48h] [rbp-38h]
  unsigned int v59; // [rsp+4Ch] [rbp-34h]

  v2 = *(_QWORD *)(a1 + 904);
  v3 = *(_QWORD *)(a1 + 920);
  if ( v3 == v2 + 48 )
  {
    *(_BYTE *)(a1 + 582) |= 8u;
    v54 = a1 + 328;
    *(_QWORD *)(a1 + 328) = 0;
LABEL_12:
    v8 = *(_QWORD *)(v2 + 112);
    v9 = v8 + 8LL * *(unsigned int *)(v2 + 120);
    v50 = (__int64 *)v9;
    if ( v8 != v9 )
    {
      v51 = *(__int64 **)(v2 + 112);
      v10 = a1;
      do
      {
        v11 = *(unsigned int **)(*v51 + 192);
        for ( i = (unsigned int *)sub_2E33140(*v51); v11 != i; i += 6 )
        {
          v13 = *(_QWORD **)(v10 + 24);
          v14 = v13[1] + 24LL * *i;
          v15 = (__int16 *)(v13[7] + 2LL * (*(_DWORD *)(v14 + 16) >> 12));
          v16 = *(_DWORD *)(v14 + 16) & 0xFFF;
          v17 = (_OWORD *)(v13[8] + 16LL * *(unsigned __int16 *)(v14 + 20));
          if ( v15 )
          {
            do
            {
              if ( (*v17 & *(_OWORD *)(i + 2)) != 0 )
              {
                v19 = *(_DWORD *)(v10 + 1208);
                v20 = *(unsigned __int16 *)(*(_QWORD *)(v10 + 1408) + 2LL * v16);
                if ( v20 >= v19 )
                  goto LABEL_28;
                v21 = *(_QWORD *)(v10 + 1200);
                while ( 1 )
                {
                  v22 = v21 + 24LL * v20;
                  if ( v16 == *(_DWORD *)(v22 + 12) )
                  {
                    v23 = *(unsigned int *)(v22 + 16);
                    if ( (_DWORD)v23 != -1 && *(_DWORD *)(v21 + 24 * v23 + 20) == -1 )
                      break;
                  }
                  v20 += 0x10000;
                  if ( v19 <= v20 )
                    goto LABEL_28;
                }
                if ( v20 == -1 )
                {
LABEL_28:
                  v55 = v17;
                  v58 = -1;
                  v57 = v54;
                  v59 = v16;
                  sub_2F932E0(v10 + 1200, (__int64)&v57);
                  v17 = v55;
                }
              }
              v18 = *v15;
              ++v17;
              ++v15;
              v16 += v18;
            }
            while ( (_WORD)v18 );
          }
        }
        LOBYTE(v9) = (_BYTE)++v51;
      }
      while ( v50 != v51 );
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 912);
    while ( v3 != v4 )
    {
      v5 = *(_WORD *)(v3 + 68);
      if ( (unsigned __int16)(v5 - 14) > 4u && v5 != 24 )
      {
        *(_QWORD *)(a1 + 328) = v3;
        v4 = v3;
        *(_BYTE *)(a1 + 582) |= 8u;
        v54 = a1 + 328;
        goto LABEL_30;
      }
      v6 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
        BUG();
      v7 = *(_QWORD *)v6;
      v3 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v3 = v7 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
            break;
          v7 = *(_QWORD *)v3;
        }
      }
    }
    *(_QWORD *)(a1 + 328) = v4;
    *(_BYTE *)(a1 + 582) |= 8u;
    v54 = a1 + 328;
    if ( !v4 )
      goto LABEL_12;
LABEL_30:
    v24 = *(_QWORD *)(v4 + 32);
    v52 = *(unsigned __int16 **)(v4 + 16);
    v25 = (_BYTE *)(v24 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF));
    v26 = (_BYTE *)(v24 + 40LL * (unsigned int)sub_2E88FE0(v4));
    if ( v25 != v26 )
    {
      while ( 1 )
      {
        v27 = (__int64)v26;
        v28 = sub_2E2FA70(v26);
        if ( v28 )
          break;
        v26 += 40;
        if ( v25 == v26 )
          goto LABEL_44;
      }
      v49 = v28;
      if ( v25 != v26 )
      {
        v53 = v4;
        do
        {
          v29 = sub_2EAB0A0(v27);
          v32 = *(unsigned int *)(v27 + 8);
          v33 = v29;
          if ( (unsigned int)(v32 - 1) <= 0x3FFFFFFE )
          {
            v39 = v49;
            if ( v52[1] <= v29 )
            {
              LODWORD(v57) = *(_DWORD *)(v27 + 8);
              v46 = &v52[20 * *v52 + 20 + *((unsigned int *)v52 + 3)];
              v47 = &v46[*((unsigned __int8 *)v52 + 8)];
              v39 = v47 != sub_2F91020(v46, (__int64)v47, (int *)&v57);
            }
            v40 = *(_QWORD *)(a1 + 24);
            v41 = *(_QWORD *)(v40 + 8);
            v42 = *(_QWORD *)(v40 + 56);
            if ( !v39 )
              v33 = -1;
            v56 = v33;
            v43 = v27;
            v44 = *(_DWORD *)(v41 + 24 * v32 + 16) & 0xFFF;
            v45 = v42 + 2LL * (*(_DWORD *)(v41 + 24 * v32 + 16) >> 12);
            do
            {
              if ( !v45 )
                break;
              v59 = v44;
              v45 += 2;
              v57 = v54;
              v58 = v56;
              sub_2F932E0(a1 + 1200, (__int64)&v57);
              v44 += *(__int16 *)(v45 - 2);
            }
            while ( *(_WORD *)(v45 - 2) );
            v27 = v43;
          }
          else if ( (int)v32 < 0 )
          {
            v38 = *(_BYTE *)(v27 + 4);
            if ( (v38 & 1) == 0
              && (v38 & 2) == 0
              && ((*(_BYTE *)(v27 + 3) & 0x10) == 0 || (*(_DWORD *)v27 & 0xFFF00) != 0) )
            {
              sub_2F91D20(a1, v54, v33, v30, v31);
            }
          }
          v34 = (_BYTE *)(v27 + 40);
          if ( v34 == v25 )
            break;
          v35 = v34;
          while ( 1 )
          {
            v27 = (__int64)v35;
            if ( (unsigned __int8)sub_2E2FA70(v35) )
              break;
            v35 += 40;
            if ( v25 == v35 )
              goto LABEL_43;
          }
        }
        while ( v25 != v35 );
LABEL_43:
        v4 = v53;
      }
    }
LABEL_44:
    v36 = *(_DWORD *)(v4 + 44);
    if ( (v36 & 4) == 0 && (v36 & 8) != 0 )
      LOBYTE(v9) = sub_2E88A90(v4, 128, 1);
    else
      LOBYTE(v9) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 7;
    if ( !(_BYTE)v9 )
    {
      v37 = *(_DWORD *)(v4 + 44);
      if ( (v37 & 4) == 0 && (v37 & 8) != 0 )
        LOBYTE(v9) = sub_2E88A90(v4, 256, 1);
      else
        v9 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 8) & 1LL;
      if ( !(_BYTE)v9 )
      {
        v2 = *(_QWORD *)(a1 + 904);
        goto LABEL_12;
      }
    }
  }
  return v9;
}
