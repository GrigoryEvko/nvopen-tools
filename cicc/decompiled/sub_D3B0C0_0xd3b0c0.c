// Function: sub_D3B0C0
// Address: 0xd3b0c0
//
__int64 __fastcall sub_D3B0C0(unsigned int *a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 i; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  unsigned int v25; // esi
  int v26; // r14d
  __int64 v27; // rdi
  _QWORD *v28; // r10
  unsigned int v29; // ecx
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rcx
  _QWORD *v35; // r15
  __int64 v36; // rax
  __int64 *v37; // rdx
  __int64 v38; // rsi
  unsigned int v39; // r8d
  unsigned __int64 v40; // r15
  char v41; // cl
  __int64 v42; // r15
  unsigned __int64 v43; // rax
  __int64 v44; // r15
  unsigned int v45; // eax
  int v46; // edx
  __int64 v47; // rdi
  unsigned int v48; // eax
  unsigned int v49; // eax
  __int64 v50; // rdi
  unsigned int v51; // ecx
  __int64 v52; // rsi
  int v53; // r9d
  _QWORD *v54; // r8
  unsigned int v55; // eax
  unsigned int v56; // eax
  __int64 v57; // rsi
  _QWORD *v58; // rdi
  unsigned int v59; // r13d
  int v60; // r8d
  __int64 v61; // rcx
  unsigned __int64 v62; // rax
  __int64 v63; // [rsp-58h] [rbp-58h]
  _QWORD *v64; // [rsp-58h] [rbp-58h]
  int v65; // [rsp-58h] [rbp-58h]
  __int64 v66; // [rsp-50h] [rbp-50h]
  __int64 v67; // [rsp-50h] [rbp-50h]
  unsigned __int64 v68; // [rsp-50h] [rbp-50h]
  __int64 *v69; // [rsp-50h] [rbp-50h]
  __int64 v70; // [rsp-48h] [rbp-48h] BYREF
  __int64 v71; // [rsp-40h] [rbp-40h]

  result = *a2;
  if ( (unsigned __int8)result > 0x1Cu && ((_BYTE)result == 61 || (_BYTE)result == 62) )
  {
    v4 = *((_QWORD *)a2 - 4);
    if ( v4 )
    {
      result = *(_QWORD *)(v4 + 8);
      if ( *(_BYTE *)(result + 8) == 14 )
      {
        v5 = *((_QWORD *)a1 + 3);
        v6 = *(_QWORD *)(*(_QWORD *)a1 + 112LL);
        if ( *(_BYTE *)v4 != 63
          || ((v33 = 4LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF), (*(_BYTE *)(v4 + 7) & 0x40) != 0)
            ? (v34 = *(_QWORD **)(v4 - 8), v64 = &v34[v33])
            : (v64 = (_QWORD *)*((_QWORD *)a2 - 4), v34 = (_QWORD *)(v4 - v33 * 8)),
              v34 == v64) )
        {
LABEL_9:
          v9 = sub_DD8400(v6, v4);
          result = *(unsigned __int16 *)(v9 + 24);
          v66 = v4;
        }
        else
        {
          v66 = *((_QWORD *)a2 - 4);
          v35 = v34;
          do
          {
            v36 = sub_DD8400(v6, *v35);
            if ( !(unsigned __int8)sub_DADE90(v6, v36, v5) )
            {
              if ( v66 != v4 )
                goto LABEL_9;
              v66 = *v35;
            }
            v35 += 4;
          }
          while ( v64 != v35 );
          v9 = sub_DD8400(v6, v66);
          result = *(unsigned __int16 *)(v9 + 24);
          if ( v66 != v4 )
          {
            for ( i = (unsigned int)(result - 2); (unsigned __int16)(result - 2) <= 2u; i = (unsigned int)(result - 2) )
            {
              v9 = *(_QWORD *)(v9 + 32);
              result = *(unsigned __int16 *)(v9 + 24);
            }
          }
        }
        if ( (_WORD)result == 8 && v5 == *(_QWORD *)(v9 + 48) )
        {
          v11 = sub_D33D80((_QWORD *)v9, v6, i, v8, v10);
          v12 = v11;
          if ( v4 == v66 && *(_WORD *)(v11 + 24) == 6 )
          {
            v37 = *(__int64 **)(v11 + 32);
            result = *v37;
            if ( *(_WORD *)(*v37 + 24) )
              return result;
            v38 = *(_QWORD *)(result + 32);
            v39 = *(_DWORD *)(v38 + 32);
            v40 = *(_QWORD *)(v38 + 24);
            result = 1LL << ((unsigned __int8)v39 - 1);
            if ( v39 > 0x40 )
            {
              v47 = v38 + 24;
              v65 = *(_DWORD *)(v38 + 32);
              v69 = v37;
              if ( (*(_QWORD *)(v40 + 8LL * ((v39 - 1) >> 6)) & result) != 0 )
                result = sub_C44500(v47);
              else
                result = sub_C444A0(v47);
              v37 = v69;
              if ( (unsigned int)(v65 + 1 - result) > 0x40 )
                return result;
              v44 = *(_QWORD *)v40;
            }
            else
            {
              if ( (result & v40) != 0 )
              {
                if ( !v39 )
                  return result;
                result = 64;
                v41 = 64 - v39;
                v42 = v40 << (64 - (unsigned __int8)v39);
                if ( v42 != -1 )
                {
                  _BitScanReverse64(&v43, ~v42);
                  result = (unsigned int)v43 ^ 0x3F;
                }
                if ( v39 + 1 - (unsigned int)result > 0x40 )
                  return result;
              }
              else
              {
                if ( v40 )
                {
                  _BitScanReverse64(&v62, v40);
                  result = v62 ^ 0x3F;
                  if ( !(_DWORD)result )
                    return result;
                }
                if ( !v39 )
                  return result;
                v41 = 64 - v39;
                v42 = v40 << (64 - (unsigned __int8)v39);
              }
              v44 = v42 >> v41;
            }
            if ( v44 != 1 )
              return result;
            v12 = v37[1];
          }
          result = sub_DADE90(v6, v12, v5);
          if ( (_BYTE)result )
          {
            result = *(unsigned __int16 *)(v12 + 24);
            if ( (_WORD)result == 15
              || (result = (unsigned int)(result - 2), (unsigned __int16)result <= 2u)
              && (result = *(_QWORD *)(v12 + 32), *(_WORD *)(result + 24) == 15) )
            {
              if ( (_BYTE)qword_4F86EA8 )
              {
                v13 = sub_DEF9D0(*(_QWORD *)a1, v12);
                v67 = sub_AA4E30(**(_QWORD **)(*((_QWORD *)a1 + 3) + 32LL));
                v14 = sub_D95540(v12);
                v63 = v67;
                v15 = sub_9208B0(v67, v14);
                v71 = v16;
                v70 = v15;
                v68 = sub_CA1930(&v70);
                v17 = sub_D95540(v13);
                v18 = sub_9208B0(v63, v17);
                v71 = v19;
                v70 = v18;
                v20 = sub_CA1930(&v70);
                v21 = *(_QWORD *)(*(_QWORD *)a1 + 112LL);
                if ( v68 > v20 )
                {
                  v32 = sub_D95540(v12);
                  v23 = v12;
                  v13 = sub_DC2B70(v21, v13, v32, 0);
                }
                else
                {
                  v22 = sub_D95540(v13);
                  v23 = sub_DD2D10(v21, v12, v22);
                }
                v24 = sub_DCC810(v21, v23, v13, 0, 0);
                result = sub_DBEDC0(v21, v24);
                if ( !(_BYTE)result )
                {
                  if ( (unsigned __int16)(*(_WORD *)(v12 + 24) - 2) <= 2u )
                    v12 = *(_QWORD *)(v12 + 32);
                  v25 = a1[36];
                  if ( v25 )
                  {
                    v26 = 1;
                    v27 = *((_QWORD *)a1 + 16);
                    v28 = 0;
                    v29 = (v25 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
                    v30 = (_QWORD *)(v27 + 16LL * v29);
                    v31 = *v30;
                    if ( *v30 == v4 )
                    {
LABEL_25:
                      result = (__int64)(v30 + 1);
LABEL_26:
                      *(_QWORD *)result = v12;
                      return result;
                    }
                    while ( v31 != -4096 )
                    {
                      if ( v31 == -8192 && !v28 )
                        v28 = v30;
                      v29 = (v25 - 1) & (v26 + v29);
                      v30 = (_QWORD *)(v27 + 16LL * v29);
                      v31 = *v30;
                      if ( *v30 == v4 )
                        goto LABEL_25;
                      ++v26;
                    }
                    if ( !v28 )
                      v28 = v30;
                    v45 = a1[34];
                    ++*((_QWORD *)a1 + 15);
                    v46 = v45 + 1;
                    if ( 4 * (v45 + 1) < 3 * v25 )
                    {
                      if ( v25 - a1[35] - v46 > v25 >> 3 )
                      {
LABEL_62:
                        a1[34] = v46;
                        if ( *v28 != -4096 )
                          --a1[35];
                        *v28 = v4;
                        result = (__int64)(v28 + 1);
                        v28[1] = 0;
                        goto LABEL_26;
                      }
                      sub_D3AEE0((__int64)(a1 + 30), v25);
                      v55 = a1[36];
                      if ( v55 )
                      {
                        v56 = v55 - 1;
                        v57 = *((_QWORD *)a1 + 16);
                        v58 = 0;
                        v59 = v56 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
                        v60 = 1;
                        v46 = a1[34] + 1;
                        v28 = (_QWORD *)(v57 + 16LL * v59);
                        v61 = *v28;
                        if ( *v28 != v4 )
                        {
                          while ( v61 != -4096 )
                          {
                            if ( v61 == -8192 && !v58 )
                              v58 = v28;
                            v59 = v56 & (v60 + v59);
                            v28 = (_QWORD *)(v57 + 16LL * v59);
                            v61 = *v28;
                            if ( *v28 == v4 )
                              goto LABEL_62;
                            ++v60;
                          }
                          if ( v58 )
                            v28 = v58;
                        }
                        goto LABEL_62;
                      }
LABEL_98:
                      ++a1[34];
                      BUG();
                    }
                  }
                  else
                  {
                    ++*((_QWORD *)a1 + 15);
                  }
                  sub_D3AEE0((__int64)(a1 + 30), 2 * v25);
                  v48 = a1[36];
                  if ( v48 )
                  {
                    v49 = v48 - 1;
                    v50 = *((_QWORD *)a1 + 16);
                    v51 = v49 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
                    v46 = a1[34] + 1;
                    v28 = (_QWORD *)(v50 + 16LL * v51);
                    v52 = *v28;
                    if ( *v28 != v4 )
                    {
                      v53 = 1;
                      v54 = 0;
                      while ( v52 != -4096 )
                      {
                        if ( !v54 && v52 == -8192 )
                          v54 = v28;
                        v51 = v49 & (v53 + v51);
                        v28 = (_QWORD *)(v50 + 16LL * v51);
                        v52 = *v28;
                        if ( *v28 == v4 )
                          goto LABEL_62;
                        ++v53;
                      }
                      if ( v54 )
                        v28 = v54;
                    }
                    goto LABEL_62;
                  }
                  goto LABEL_98;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
