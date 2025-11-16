// Function: sub_1360180
// Address: 0x1360180
//
__int64 __fastcall sub_1360180(__int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  int v5; // r12d
  unsigned int v6; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r12
  __int64 v39; // rsi
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r14
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r14
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // r14
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // [rsp+8h] [rbp-48h]
  __int64 v88; // [rsp+8h] [rbp-48h]
  __int64 v89; // [rsp+8h] [rbp-48h]
  __int64 v90; // [rsp+8h] [rbp-48h]
  _QWORD v91[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (a2 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  v5 = (a2 >> 2) & 1;
  if ( !v5 )
  {
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 36) )
      return 4;
    if ( *(char *)(v3 + 23) >= 0 )
      goto LABEL_150;
    v17 = sub_1648A40(v3);
    v19 = v17 + v18;
    v20 = 0;
    if ( *(char *)(v3 + 23) < 0 )
      v20 = sub_1648A40(v3);
    if ( !(unsigned int)((v19 - v20) >> 4) )
    {
LABEL_150:
      v21 = *(_QWORD *)(v3 - 72);
      if ( !*(_BYTE *)(v21 + 16) )
      {
        v91[0] = *(_QWORD *)(v21 + 112);
        if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 36) )
          return 4;
      }
    }
    if ( !(unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 36) )
    {
      if ( *(char *)(v3 + 23) < 0 )
      {
        v41 = sub_1648A40(v3);
        v43 = v41 + v42;
        v44 = 0;
        if ( *(char *)(v3 + 23) < 0 )
          v44 = sub_1648A40(v3);
        if ( (unsigned int)((v43 - v44) >> 4) )
          goto LABEL_151;
      }
      v45 = *(_QWORD *)(v3 - 72);
      if ( *(_BYTE *)(v45 + 16)
        || (v91[0] = *(_QWORD *)(v45 + 112), !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 36)) )
      {
LABEL_151:
        if ( !(unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 37) )
        {
          if ( *(char *)(v3 + 23) < 0 )
          {
            v46 = sub_1648A40(v3);
            v48 = v46 + v47;
            v49 = *(char *)(v3 + 23) >= 0 ? 0LL : sub_1648A40(v3);
            if ( v49 != v48 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v49 + 8LL) <= 1u )
              {
                v49 += 16;
                if ( v48 == v49 )
                  goto LABEL_127;
              }
              goto LABEL_72;
            }
          }
LABEL_127:
          v84 = *(_QWORD *)(v3 - 72);
          if ( *(_BYTE *)(v84 + 16)
            || (v91[0] = *(_QWORD *)(v84 + 112), !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 37)) )
          {
LABEL_72:
            if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 36) )
              goto LABEL_79;
            if ( *(char *)(v3 + 23) >= 0 )
              goto LABEL_152;
            v50 = sub_1648A40(v3);
            v52 = v50 + v51;
            v53 = 0;
            if ( *(char *)(v3 + 23) < 0 )
              v53 = sub_1648A40(v3);
            if ( !(unsigned int)((v52 - v53) >> 4) )
            {
LABEL_152:
              v54 = *(_QWORD *)(v3 - 72);
              if ( !*(_BYTE *)(v54 + 16) )
              {
                v91[0] = *(_QWORD *)(v54 + 112);
                if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 36) )
                  goto LABEL_79;
              }
            }
            if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 57)
              || (v85 = *(_QWORD *)(v3 - 72), !*(_BYTE *)(v85 + 16))
              && (v91[0] = *(_QWORD *)(v85 + 112), (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 57)) )
            {
LABEL_79:
              v6 = 62;
            }
            else
            {
              v6 = 63;
            }
LABEL_30:
            if ( !(unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 4) )
            {
              if ( *(char *)(v3 + 23) < 0 )
              {
                v22 = sub_1648A40(v3);
                v24 = v23 + v22;
                v25 = 0;
                v87 = v24;
                if ( *(char *)(v3 + 23) < 0 )
                  v25 = sub_1648A40(v3);
                if ( (unsigned int)((v87 - v25) >> 4) )
                  goto LABEL_153;
              }
              v26 = *(_QWORD *)(v3 - 72);
              if ( *(_BYTE *)(v26 + 16)
                || (v91[0] = *(_QWORD *)(v26 + 112), !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 4)) )
              {
LABEL_153:
                if ( !(unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 13) )
                {
                  if ( *(char *)(v3 + 23) < 0 )
                  {
                    v27 = sub_1648A40(v3);
                    v29 = v28 + v27;
                    v30 = 0;
                    v88 = v29;
                    if ( *(char *)(v3 + 23) < 0 )
                      v30 = sub_1648A40(v3);
                    if ( (unsigned int)((v88 - v30) >> 4) )
                      goto LABEL_154;
                  }
                  v31 = *(_QWORD *)(v3 - 72);
                  if ( *(_BYTE *)(v31 + 16)
                    || (v91[0] = *(_QWORD *)(v31 + 112), !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 13)) )
                  {
LABEL_154:
                    if ( !(unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 14) )
                    {
                      if ( *(char *)(v3 + 23) >= 0 )
                      {
                        v35 = *(_QWORD *)(v3 - 72);
                        if ( *(_BYTE *)(v35 + 16) )
                          goto LABEL_55;
                      }
                      else
                      {
                        v32 = sub_1648A40(v3);
                        v34 = v32 + v33;
                        if ( *(char *)(v3 + 23) >= 0 )
                        {
                          if ( (unsigned int)(v34 >> 4) )
                            goto LABEL_55;
                        }
                        else if ( (unsigned int)((v34 - sub_1648A40(v3)) >> 4) )
                        {
                          goto LABEL_51;
                        }
                        v35 = *(_QWORD *)(v3 - 72);
                        if ( *(_BYTE *)(v35 + 16) )
                          goto LABEL_51;
                      }
                      v91[0] = *(_QWORD *)(v35 + 112);
                      if ( !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 14) )
                        goto LABEL_51;
                    }
                    goto LABEL_50;
                  }
                }
                goto LABEL_120;
              }
            }
LABEL_14:
            v6 &= 0xFu;
            goto LABEL_15;
          }
        }
      }
    }
    v6 = 61;
    goto LABEL_30;
  }
  if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 36) )
    return 4;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_155;
  v8 = sub_1648A40(v3);
  v10 = v8 + v9;
  v11 = 0;
  if ( *(char *)(v3 + 23) < 0 )
    v11 = sub_1648A40(v3);
  if ( !(unsigned int)((v10 - v11) >> 4) )
  {
LABEL_155:
    v12 = *(_QWORD *)(v3 - 24);
    if ( !*(_BYTE *)(v12 + 16) )
    {
      v91[0] = *(_QWORD *)(v12 + 112);
      if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 36) )
        return 4;
    }
  }
  if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 36) )
    goto LABEL_12;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_156;
  v69 = sub_1648A40(v3);
  v71 = v69 + v70;
  v72 = 0;
  if ( *(char *)(v3 + 23) < 0 )
    v72 = sub_1648A40(v3);
  if ( !(unsigned int)((v71 - v72) >> 4) )
  {
LABEL_156:
    v73 = *(_QWORD *)(v3 - 24);
    if ( !*(_BYTE *)(v73 + 16) )
    {
      v91[0] = *(_QWORD *)(v73 + 112);
      if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 36) )
        goto LABEL_12;
    }
  }
  if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 37) )
    goto LABEL_12;
  if ( *(char *)(v3 + 23) < 0 )
  {
    v74 = sub_1648A40(v3);
    v76 = v74 + v75;
    v77 = *(char *)(v3 + 23) >= 0 ? 0LL : sub_1648A40(v3);
    if ( v77 != v76 )
    {
      while ( *(_DWORD *)(*(_QWORD *)v77 + 8LL) <= 1u )
      {
        v77 += 16;
        if ( v76 == v77 )
          goto LABEL_123;
      }
      goto LABEL_112;
    }
  }
LABEL_123:
  v83 = *(_QWORD *)(v3 - 24);
  if ( !*(_BYTE *)(v83 + 16) )
  {
    v91[0] = *(_QWORD *)(v83 + 112);
    if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 37) )
    {
LABEL_12:
      v6 = 61;
      goto LABEL_13;
    }
  }
LABEL_112:
  if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 36) )
    goto LABEL_119;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_157;
  v78 = sub_1648A40(v3);
  v80 = v78 + v79;
  v81 = 0;
  if ( *(char *)(v3 + 23) < 0 )
    v81 = sub_1648A40(v3);
  if ( !(unsigned int)((v80 - v81) >> 4) )
  {
LABEL_157:
    v82 = *(_QWORD *)(v3 - 24);
    if ( !*(_BYTE *)(v82 + 16) )
    {
      v91[0] = *(_QWORD *)(v82 + 112);
      if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 36) )
        goto LABEL_119;
    }
  }
  if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 57)
    || (v86 = *(_QWORD *)(v3 - 24), !*(_BYTE *)(v86 + 16))
    && (v91[0] = *(_QWORD *)(v86 + 112), (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 57)) )
  {
LABEL_119:
    v6 = 62;
  }
  else
  {
    v6 = 63;
  }
LABEL_13:
  if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 4) )
    goto LABEL_14;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_158;
  v55 = sub_1648A40(v3);
  v57 = v56 + v55;
  v58 = 0;
  v89 = v57;
  if ( *(char *)(v3 + 23) < 0 )
    v58 = sub_1648A40(v3);
  if ( !(unsigned int)((v89 - v58) >> 4) )
  {
LABEL_158:
    v59 = *(_QWORD *)(v3 - 24);
    if ( !*(_BYTE *)(v59 + 16) )
    {
      v91[0] = *(_QWORD *)(v59 + 112);
      if ( (unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 4) )
        goto LABEL_14;
    }
  }
  if ( !(unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 13) )
  {
    if ( *(char *)(v3 + 23) < 0 )
    {
      v60 = sub_1648A40(v3);
      v62 = v61 + v60;
      v63 = 0;
      v90 = v62;
      if ( *(char *)(v3 + 23) < 0 )
        v63 = sub_1648A40(v3);
      if ( (unsigned int)((v90 - v63) >> 4) )
        goto LABEL_159;
    }
    v64 = *(_QWORD *)(v3 - 24);
    if ( *(_BYTE *)(v64 + 16) || (v91[0] = *(_QWORD *)(v64 + 112), !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 13)) )
    {
LABEL_159:
      if ( (unsigned __int8)sub_1560260(v3 + 56, 0xFFFFFFFFLL, 14) )
      {
LABEL_50:
        v6 &= 0x1Fu;
        goto LABEL_15;
      }
      if ( *(char *)(v3 + 23) >= 0 )
      {
        v68 = *(_QWORD *)(v3 - 24);
        if ( !*(_BYTE *)(v68 + 16) )
        {
LABEL_98:
          v91[0] = *(_QWORD *)(v68 + 112);
          if ( !(unsigned __int8)sub_1560260(v91, 0xFFFFFFFFLL, 14) )
            goto LABEL_16;
          goto LABEL_50;
        }
      }
      else
      {
        v65 = sub_1648A40(v3);
        v67 = v65 + v66;
        if ( *(char *)(v3 + 23) < 0 )
        {
          if ( (unsigned int)((v67 - sub_1648A40(v3)) >> 4) )
            goto LABEL_16;
          goto LABEL_97;
        }
        if ( !(unsigned int)(v67 >> 4) )
        {
LABEL_97:
          v68 = *(_QWORD *)(v3 - 24);
          if ( *(_BYTE *)(v68 + 16) )
            goto LABEL_16;
          goto LABEL_98;
        }
      }
LABEL_20:
      v16 = (__int64 *)(v3 - 24);
      goto LABEL_56;
    }
  }
LABEL_120:
  v6 &= 0x17u;
LABEL_15:
  if ( !(_BYTE)v5 )
  {
LABEL_51:
    if ( *(char *)(v3 + 23) < 0 )
    {
      v36 = sub_1648A40(v3);
      v38 = v36 + v37;
      if ( *(char *)(v3 + 23) < 0 )
        v38 -= sub_1648A40(v3);
      if ( (unsigned int)(v38 >> 4) )
        return v6;
    }
LABEL_55:
    v16 = (__int64 *)(v3 - 72);
LABEL_56:
    v39 = *v16;
    if ( !*(_BYTE *)(*v16 + 16) )
    {
      if ( *a1 )
        v40 = sub_134CE70(*a1, v39);
      else
        v40 = sub_1360090((__int64)a1, v39);
      v6 &= v40;
    }
    return v6;
  }
LABEL_16:
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_20;
  v13 = sub_1648A40(v3);
  v15 = v13 + v14;
  if ( *(char *)(v3 + 23) < 0 )
    v15 -= sub_1648A40(v3);
  if ( !(unsigned int)(v15 >> 4) )
    goto LABEL_20;
  return v6;
}
