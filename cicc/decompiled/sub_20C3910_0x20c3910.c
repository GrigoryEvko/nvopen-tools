// Function: sub_20C3910
// Address: 0x20c3910
//
__int64 __fastcall sub_20C3910(__int64 *a1, __int64 a2, int a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  int v8; // r14d
  unsigned int v9; // esi
  unsigned int v10; // r13d
  __int64 v11; // r12
  __int64 v12; // rbx
  unsigned int v13; // r13d
  __int16 v14; // ax
  __int64 v15; // rax
  _QWORD *v16; // rsi
  unsigned int v17; // edx
  __int16 v18; // ax
  _WORD *v19; // rdx
  _WORD *v20; // rdi
  unsigned __int16 v21; // cx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int16 v25; // ax
  __int16 v26; // cx
  __int16 v27; // ax
  __int64 v28; // r14
  __int64 v29; // r12
  _QWORD *v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 v34; // r15
  unsigned int v35; // edx
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // r11
  __int16 v40; // r10
  unsigned int v41; // esi
  __int16 v42; // dx
  _WORD *v43; // rdi
  _WORD *v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // r8
  __int64 v47; // rdx
  __int64 v48; // r14
  __int16 v49; // r12
  __int16 *v50; // rsi
  __int16 v51; // ax
  __int16 *v52; // rsi
  unsigned __int16 v53; // dx
  __int16 v54; // cx
  __int16 *v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rax
  __int16 v58; // ax
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 (*v61)(); // rax
  __int16 v62; // dx
  __int64 v63; // [rsp+10h] [rbp-A0h]
  __int64 v64; // [rsp+10h] [rbp-A0h]
  __int64 v67; // [rsp+20h] [rbp-90h]
  __int64 v69; // [rsp+30h] [rbp-80h]
  __int64 v70; // [rsp+38h] [rbp-78h]
  unsigned int v71; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v72; // [rsp+48h] [rbp-68h]
  __int64 v73; // [rsp+50h] [rbp-60h]
  unsigned __int16 v74; // [rsp+58h] [rbp-58h]
  _WORD *v75; // [rsp+60h] [rbp-50h]
  int v76; // [rsp+68h] [rbp-48h]
  unsigned __int16 v77; // [rsp+70h] [rbp-40h]
  __int64 v78; // [rsp+78h] [rbp-38h]

  v63 = a1[9];
  result = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)result )
  {
    v6 = 0;
    v7 = 40 * result;
    v8 = a3 + 1;
    do
    {
      result = v6 + *(_QWORD *)(a2 + 32);
      if ( !*(_BYTE *)result && (*(_BYTE *)(result + 3) & 0x10) != 0 )
      {
        v9 = *(_DWORD *)(result + 8);
        if ( v9 )
          result = (__int64)sub_20C35E0((__int64)a1, v9, v8);
      }
      v6 += 40;
    }
    while ( v7 != v6 );
    v10 = *(_DWORD *)(a2 + 40);
    if ( v10 )
    {
      v11 = 0;
      v69 = v10;
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(a2 + 32) + 40 * v11;
          if ( !*(_BYTE *)v12 && (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
          {
            v13 = *(_DWORD *)(v12 + 8);
            if ( v13 )
              break;
          }
          if ( v69 == ++v11 )
            goto LABEL_43;
        }
        v14 = *(_WORD *)(a2 + 46);
        if ( (v14 & 4) != 0 || (v14 & 8) == 0 )
          v15 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 4) & 1LL;
        else
          LOBYTE(v15) = sub_1E15D00(a2, 0x10u, 1);
        if ( (_BYTE)v15
          || ((v58 = *(_WORD *)(a2 + 46), (v58 & 4) == 0) && (v58 & 8) != 0
            ? (LOBYTE(v59) = sub_1E15D00(a2, 0x10000000u, 1))
            : (v59 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 28) & 1LL),
              (_BYTE)v59
           || (v60 = a1[3], v61 = *(__int64 (**)())(*(_QWORD *)v60 + 656LL), v61 != sub_1D918C0)
           && ((unsigned __int8 (__fastcall *)(__int64, __int64))v61)(v60, a2)
           || **(_WORD **)(a2 + 16) == 1) )
        {
          sub_20C2470((_QWORD *)a1[9], v13, 0);
        }
        v71 = v13;
        v16 = (_QWORD *)a1[4];
        if ( !v16 )
        {
          v72 = 0;
          LOBYTE(v73) = 0;
          v74 = 0;
          v75 = 0;
          v76 = 0;
          v77 = 0;
          v78 = 0;
          BUG();
        }
        v75 = 0;
        v72 = v16 + 1;
        v74 = 0;
        v77 = 0;
        LOBYTE(v73) = 0;
        v76 = 0;
        v78 = 0;
        v17 = *(_DWORD *)(v16[1] + 24LL * v13 + 16);
        v18 = v13 * (v17 & 0xF);
        v19 = (_WORD *)(v16[7] + 2LL * (v17 >> 4));
        v20 = v19 + 1;
        v74 = *v19 + v18;
        v75 = v19 + 1;
        while ( v20 )
        {
          v76 = *(_DWORD *)(v16[6] + 4LL * v74);
          v21 = v76;
          if ( (_WORD)v76 )
          {
            while ( 2 )
            {
              v22 = *(unsigned int *)(v16[1] + 24LL * v21 + 8);
              v23 = v16[7];
              v77 = v21;
              v24 = v23 + 2 * v22;
              v78 = v24;
              while ( v24 )
              {
                v25 = v77;
                if ( v13 != v77 )
                {
                  while ( 1 )
                  {
                    v30 = (_QWORD *)a1[9];
                    if ( *(_DWORD *)(v30[13] + 4LL * v25) != -1 && *(_DWORD *)(v30[16] + 4LL * v25) == -1 )
                      sub_20C2470(v30, v13, v25);
                    sub_1E1D5E0((__int64)&v71);
                    if ( !v75 )
                      break;
                    v25 = v77;
                  }
                  goto LABEL_40;
                }
                v24 += 2;
                v78 = v24;
                v26 = *(_WORD *)(v24 - 2);
                v77 += v26;
                if ( !v26 )
                {
                  v78 = 0;
                  break;
                }
              }
              v21 = HIWORD(v76);
              v76 = HIWORD(v76);
              if ( v21 )
                continue;
              break;
            }
          }
          v75 = ++v20;
          v27 = *(v20 - 1);
          v74 += v27;
          if ( !v27 )
          {
            v75 = 0;
            break;
          }
        }
LABEL_40:
        v31 = *(_QWORD *)(a2 + 16);
        v32 = 0;
        if ( *(unsigned __int16 *)(v31 + 2) > (unsigned int)v11 )
          v32 = sub_1F3AD60(a1[3], v31, v11, (_QWORD *)a1[4], a1[1]);
        v71 = v13;
        ++v11;
        v72 = (_QWORD *)v12;
        v73 = v32;
        sub_20C33D0(v63 + 56, (int *)&v71);
      }
      while ( v69 != v11 );
LABEL_43:
      result = *(unsigned int *)(a2 + 40);
      v33 = a1;
      if ( (_DWORD)result )
      {
        v29 = a2;
        v34 = v63;
        v28 = 0;
        v70 = 40 * result;
        do
        {
          result = v28 + *(_QWORD *)(v29 + 32);
          if ( !*(_BYTE *)result && (*(_BYTE *)(result + 3) & 0x10) != 0 )
          {
            v35 = *(_DWORD *)(result + 8);
            if ( v35 )
            {
              result = *(_QWORD *)(v29 + 16);
              if ( *(_WORD *)result != 6 )
              {
                result = *(_QWORD *)(a4 + 16);
                if ( !result )
                  goto LABEL_56;
                v36 = a4 + 8;
                do
                {
                  while ( 1 )
                  {
                    v37 = *(_QWORD *)(result + 16);
                    v38 = *(_QWORD *)(result + 24);
                    if ( v35 <= *(_DWORD *)(result + 32) )
                      break;
                    result = *(_QWORD *)(result + 24);
                    if ( !v38 )
                      goto LABEL_54;
                  }
                  v36 = result;
                  result = *(_QWORD *)(result + 16);
                }
                while ( v37 );
LABEL_54:
                if ( a4 + 8 == v36 || v35 < *(_DWORD *)(v36 + 32) )
                {
LABEL_56:
                  result = v33[4];
                  if ( !result )
                  {
                    v71 = v35;
                    v72 = 0;
                    LOBYTE(v73) = 1;
                    v74 = 0;
                    v75 = 0;
                    v76 = 0;
                    v77 = 0;
                    v78 = 0;
                    BUG();
                  }
                  v71 = v35;
                  v72 = (_QWORD *)(result + 8);
                  v74 = 0;
                  v75 = 0;
                  v77 = 0;
                  v39 = 24LL * v35;
                  v40 = v35;
                  LOBYTE(v73) = 1;
                  v76 = 0;
                  v78 = 0;
                  v41 = *(_DWORD *)(*(_QWORD *)(result + 8) + v39 + 16);
                  v42 = (v41 & 0xF) * v35;
                  v43 = (_WORD *)(*(_QWORD *)(result + 56) + 2LL * (v41 >> 4));
                  v44 = v43 + 1;
                  v74 = *v43 + v42;
                  v75 = v43 + 1;
                  while ( v44 )
                  {
                    v76 = *(_DWORD *)(*(_QWORD *)(result + 48) + 4LL * v74);
                    v45 = (unsigned __int16)v76;
                    if ( (_WORD)v76 )
                    {
                      while ( 1 )
                      {
                        v46 = *(unsigned int *)(*(_QWORD *)(result + 8) + 24LL * (unsigned __int16)v45 + 8);
                        v47 = *(_QWORD *)(result + 56);
                        v77 = v45;
                        v78 = v47 + 2 * v46;
                        if ( v78 )
                          break;
                        v45 = HIWORD(v76);
                        v76 = HIWORD(v76);
                        if ( !(_WORD)v45 )
                          goto LABEL_89;
                      }
                      v67 = v28;
                      v48 = v39;
                      v64 = v29;
                      v49 = v40;
                      while ( 1 )
                      {
                        v50 = (__int16 *)(*(_QWORD *)(result + 56)
                                        + 2LL * *(unsigned int *)(*(_QWORD *)(result + 8) + v48 + 8));
                        v51 = *v50;
                        v52 = v50 + 1;
                        v53 = v49 + v51;
                        if ( !v51 )
                          v52 = 0;
LABEL_67:
                        v55 = v52;
                        while ( 1 )
                        {
                          if ( !v55 )
                          {
                            v56 = 4 * v45;
LABEL_70:
                            *(_DWORD *)(*(_QWORD *)(v34 + 128) + v56) = a3;
                            goto LABEL_71;
                          }
                          if ( v53 == (_WORD)v45 )
                            break;
                          v54 = *v55;
                          v52 = 0;
                          ++v55;
                          v53 += v54;
                          if ( !v54 )
                            goto LABEL_67;
                        }
                        v57 = v33[9];
                        v56 = 4LL * v53;
                        if ( *(_DWORD *)(*(_QWORD *)(v57 + 104) + v56) == -1
                          || *(_DWORD *)(*(_QWORD *)(v57 + 128) + 4LL * v53) != -1 )
                        {
                          goto LABEL_70;
                        }
LABEL_71:
                        result = sub_1E1D5E0((__int64)&v71);
                        if ( !v75 )
                          break;
                        result = v33[4];
                        v45 = v77;
                      }
                      v28 = v67;
                      v29 = v64;
                      break;
                    }
LABEL_89:
                    v75 = ++v44;
                    v62 = *(v44 - 1);
                    v74 += v62;
                    if ( !v62 )
                    {
                      v75 = 0;
                      break;
                    }
                  }
                }
              }
            }
          }
          v28 += 40;
        }
        while ( v70 != v28 );
      }
    }
  }
  return result;
}
