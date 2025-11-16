// Function: sub_EF9210
// Address: 0xef9210
//
_BOOL8 __fastcall sub_EF9210(_QWORD *a1)
{
  _QWORD *v1; // r14
  __int64 v2; // r15
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // ecx
  _QWORD *v8; // r14
  char v9; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned int v14; // esi
  _QWORD *v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // esi
  _QWORD *v21; // r14
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // ecx
  _QWORD *v27; // r14
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned int v32; // esi
  _QWORD *v33; // r13
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned int v38; // ecx
  _QWORD *v39; // r12
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r13
  unsigned int v44; // eax
  _QWORD *v45; // r14
  _QWORD *v46; // r12
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rbx
  unsigned int v51; // eax
  _QWORD *v52; // r9
  _QWORD *v53; // r13
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  unsigned int v58; // eax
  __int64 v59; // r15
  __int64 v60; // r14
  __int64 v61; // [rsp+0h] [rbp-F0h]
  _QWORD *v62; // [rsp+8h] [rbp-E8h]
  _QWORD *v63; // [rsp+10h] [rbp-E0h]
  _QWORD *v64; // [rsp+18h] [rbp-D8h]
  _QWORD *v65; // [rsp+20h] [rbp-D0h]
  _QWORD *v66; // [rsp+28h] [rbp-C8h]
  _QWORD *v67; // [rsp+30h] [rbp-C0h]
  _QWORD *v68; // [rsp+38h] [rbp-B8h]
  char v69; // [rsp+47h] [rbp-A9h]
  _QWORD *v70; // [rsp+48h] [rbp-A8h]
  _QWORD *v71; // [rsp+50h] [rbp-A0h]
  _QWORD *v72; // [rsp+58h] [rbp-98h]
  __int64 v73; // [rsp+68h] [rbp-88h]
  _QWORD *v74; // [rsp+70h] [rbp-80h]
  __int64 v75; // [rsp+78h] [rbp-78h]
  _QWORD *v76; // [rsp+80h] [rbp-70h]
  __int64 v77; // [rsp+88h] [rbp-68h]
  _QWORD *v78; // [rsp+90h] [rbp-60h]
  __int64 v79; // [rsp+98h] [rbp-58h]
  _QWORD *v80; // [rsp+A0h] [rbp-50h]
  __int64 v81; // [rsp+A8h] [rbp-48h]
  _QWORD *v82; // [rsp+B0h] [rbp-40h]
  __int64 v83; // [rsp+B8h] [rbp-38h]

  v1 = a1;
  if ( unk_4F838D3 )
  {
    v2 = a1[8];
    if ( v2 )
      return v2;
  }
  v4 = a1[20];
  if ( !a1[14] )
  {
    if ( !v4 )
      return v1[7] != 0;
    v6 = a1[18];
    goto LABEL_10;
  }
  v5 = a1[12];
  if ( v4 )
  {
    v6 = a1[18];
    v7 = *(_DWORD *)(v5 + 32);
    if ( *(_DWORD *)(v6 + 32) <= v7 && (*(_DWORD *)(v6 + 32) != v7 || *(_DWORD *)(v6 + 36) <= *(_DWORD *)(v5 + 36)) )
    {
LABEL_10:
      v82 = (_QWORD *)(v6 + 48);
      if ( v6 + 48 == *(_QWORD *)(v6 + 64) )
        return v1[7] != 0;
      v8 = *(_QWORD **)(v6 + 64);
      v83 = 0;
      v9 = unk_4F838D3;
      while ( 1 )
      {
        if ( v9 )
        {
          v10 = v8[14];
          if ( v10 )
            goto LABEL_41;
        }
        v11 = v8[26];
        if ( v8[20] )
        {
          v12 = v8[18];
          if ( !v11
            || (v13 = v8[24], v14 = *(_DWORD *)(v13 + 32), *(_DWORD *)(v12 + 32) < v14)
            || *(_DWORD *)(v12 + 32) == v14 && *(_DWORD *)(v12 + 36) < *(_DWORD *)(v13 + 36) )
          {
            v10 = *(_QWORD *)(v12 + 40);
LABEL_40:
            if ( v10 )
              goto LABEL_41;
            goto LABEL_47;
          }
        }
        else
        {
          if ( !v11 )
            goto LABEL_47;
          v13 = v8[24];
        }
        v80 = (_QWORD *)(v13 + 48);
        if ( *(_QWORD *)(v13 + 64) != v13 + 48 )
        {
          v71 = v8;
          v15 = *(_QWORD **)(v13 + 64);
          v81 = 0;
          while ( 2 )
          {
            if ( v9 )
            {
              v16 = v15[14];
              if ( v16 )
                goto LABEL_38;
            }
            v17 = v15[26];
            if ( v15[20] )
            {
              v18 = v15[18];
              if ( !v17
                || (v19 = v15[24], v20 = *(_DWORD *)(v19 + 32), *(_DWORD *)(v18 + 32) < v20)
                || *(_DWORD *)(v18 + 32) == v20 && *(_DWORD *)(v18 + 36) < *(_DWORD *)(v19 + 36) )
              {
                v16 = *(_QWORD *)(v18 + 40);
LABEL_37:
                if ( v16 )
                  goto LABEL_38;
LABEL_48:
                v16 = v15[13] != 0;
LABEL_38:
                v81 += v16;
                v15 = (_QWORD *)sub_220EF30(v15);
                if ( v80 == v15 )
                {
                  v8 = v71;
                  v10 = v81;
                  goto LABEL_40;
                }
                continue;
              }
            }
            else
            {
              if ( !v17 )
                goto LABEL_48;
              v19 = v15[24];
            }
            break;
          }
          v78 = (_QWORD *)(v19 + 48);
          if ( *(_QWORD *)(v19 + 64) == v19 + 48 )
            goto LABEL_48;
          v68 = v15;
          v21 = *(_QWORD **)(v19 + 64);
          v79 = 0;
          while ( 2 )
          {
            if ( v9 )
            {
              v22 = v21[14];
              if ( v22 )
                goto LABEL_35;
            }
            v23 = v21[26];
            if ( v21[20] )
            {
              v24 = v21[18];
              if ( !v23
                || (v25 = v21[24], v26 = *(_DWORD *)(v25 + 32), *(_DWORD *)(v24 + 32) < v26)
                || *(_DWORD *)(v24 + 32) == v26 && *(_DWORD *)(v24 + 36) < *(_DWORD *)(v25 + 36) )
              {
                v22 = *(_QWORD *)(v24 + 40);
                goto LABEL_34;
              }
            }
            else
            {
              if ( !v23 )
                goto LABEL_51;
              v25 = v21[24];
            }
            v76 = (_QWORD *)(v25 + 48);
            if ( *(_QWORD *)(v25 + 64) == v25 + 48 )
            {
LABEL_51:
              v22 = v21[13] != 0;
              goto LABEL_35;
            }
            v66 = v21;
            v27 = *(_QWORD **)(v25 + 64);
            v77 = 0;
LABEL_60:
            if ( v9 )
            {
              v28 = v27[14];
              if ( v28 )
                goto LABEL_67;
            }
            v29 = v27[26];
            if ( v27[20] )
            {
              v30 = v27[18];
              if ( !v29
                || (v31 = v27[24], v32 = *(_DWORD *)(v31 + 32), *(_DWORD *)(v30 + 32) < v32)
                || *(_DWORD *)(v30 + 32) == v32 && *(_DWORD *)(v30 + 36) < *(_DWORD *)(v31 + 36) )
              {
                v28 = *(_QWORD *)(v30 + 40);
                goto LABEL_66;
              }
            }
            else
            {
              if ( !v29 )
                goto LABEL_69;
              v31 = v27[24];
            }
            v33 = *(_QWORD **)(v31 + 64);
            v74 = (_QWORD *)(v31 + 48);
            if ( v33 == (_QWORD *)(v31 + 48) )
            {
LABEL_69:
              v28 = v27[13] != 0;
              goto LABEL_67;
            }
            v65 = v27;
            v75 = 0;
LABEL_76:
            if ( v9 )
            {
              v34 = v33[14];
              if ( v34 )
                goto LABEL_83;
            }
            v35 = v33[26];
            if ( v33[20] )
            {
              v36 = v33[18];
              if ( !v35
                || (v37 = v33[24], v38 = *(_DWORD *)(v37 + 32), *(_DWORD *)(v36 + 32) < v38)
                || *(_DWORD *)(v36 + 32) == v38 && *(_DWORD *)(v36 + 36) < *(_DWORD *)(v37 + 36) )
              {
                v34 = *(_QWORD *)(v36 + 40);
                goto LABEL_82;
              }
            }
            else
            {
              if ( !v35 )
                goto LABEL_86;
              v37 = v33[24];
            }
            v39 = *(_QWORD **)(v37 + 64);
            v72 = (_QWORD *)(v37 + 48);
            if ( v39 == (_QWORD *)(v37 + 48) )
            {
LABEL_86:
              v34 = v33[13] != 0;
              goto LABEL_83;
            }
            v64 = v33;
            v73 = 0;
LABEL_91:
            if ( v9 )
            {
              v40 = v39[14];
              if ( v40 )
              {
LABEL_123:
                v73 += v40;
                v39 = (_QWORD *)sub_220EF30(v39);
                if ( v72 == v39 )
                {
                  v33 = v64;
                  v34 = v73;
LABEL_82:
                  if ( !v34 )
                    goto LABEL_86;
LABEL_83:
                  v75 += v34;
                  v33 = (_QWORD *)sub_220EF30(v33);
                  if ( v74 == v33 )
                  {
                    v27 = v65;
                    v28 = v75;
LABEL_66:
                    if ( !v28 )
                      goto LABEL_69;
LABEL_67:
                    v77 += v28;
                    v27 = (_QWORD *)sub_220EF30(v27);
                    if ( v76 == v27 )
                    {
                      v21 = v66;
                      v22 = v77;
LABEL_34:
                      if ( !v22 )
                        goto LABEL_51;
LABEL_35:
                      v79 += v22;
                      v21 = (_QWORD *)sub_220EF30(v21);
                      if ( v78 == v21 )
                      {
                        v15 = v68;
                        v16 = v79;
                        goto LABEL_37;
                      }
                      continue;
                    }
                    goto LABEL_60;
                  }
                  goto LABEL_76;
                }
                goto LABEL_91;
              }
            }
            break;
          }
          v41 = v39[26];
          if ( v39[20] )
          {
            v42 = v39[18];
            if ( !v41
              || (v43 = v39[24], v44 = *(_DWORD *)(v43 + 32), *(_DWORD *)(v42 + 32) < v44)
              || *(_DWORD *)(v42 + 32) == v44 && *(_DWORD *)(v42 + 36) < *(_DWORD *)(v43 + 36) )
            {
              v40 = *(_QWORD *)(v42 + 40);
LABEL_122:
              if ( v40 )
                goto LABEL_123;
LABEL_134:
              v40 = v39[13] != 0;
              goto LABEL_123;
            }
          }
          else
          {
            if ( !v41 )
              goto LABEL_134;
            v43 = v39[24];
          }
          v45 = *(_QWORD **)(v43 + 64);
          v70 = (_QWORD *)(v43 + 48);
          if ( v45 == (_QWORD *)(v43 + 48) )
            goto LABEL_134;
          v63 = v39;
          v40 = 0;
          v46 = v45;
          while ( 2 )
          {
            if ( v9 )
            {
              v47 = v46[14];
              if ( v47 )
              {
LABEL_120:
                v40 += v47;
                v46 = (_QWORD *)sub_220EF30(v46);
                if ( v70 == v46 )
                {
                  v39 = v63;
                  goto LABEL_122;
                }
                continue;
              }
            }
            break;
          }
          v48 = v46[26];
          if ( v46[20] )
          {
            v49 = v46[18];
            if ( !v48
              || (v50 = v46[24], v51 = *(_DWORD *)(v50 + 32), *(_DWORD *)(v49 + 32) < v51)
              || *(_DWORD *)(v49 + 32) == v51 && *(_DWORD *)(v49 + 36) < *(_DWORD *)(v50 + 36) )
            {
              v47 = *(_QWORD *)(v49 + 40);
LABEL_119:
              if ( v47 )
                goto LABEL_120;
LABEL_135:
              v47 = v46[13] != 0;
              goto LABEL_120;
            }
          }
          else
          {
            if ( !v48 )
              goto LABEL_135;
            v50 = v46[24];
          }
          v52 = *(_QWORD **)(v50 + 64);
          v67 = (_QWORD *)(v50 + 48);
          if ( v52 == (_QWORD *)(v50 + 48) )
            goto LABEL_135;
          v69 = v9;
          v47 = 0;
          v62 = v46;
          v61 = v40;
          v53 = v52;
          while ( 2 )
          {
            if ( v69 )
            {
              v54 = v53[14];
              if ( v54 )
                goto LABEL_117;
            }
            v55 = v53[26];
            if ( v53[20] )
            {
              v56 = v53[18];
              if ( !v55
                || (v57 = v53[24], v58 = *(_DWORD *)(v57 + 32), *(_DWORD *)(v56 + 32) < v58)
                || *(_DWORD *)(v56 + 32) == v58 && *(_DWORD *)(v56 + 36) < *(_DWORD *)(v57 + 36) )
              {
                v54 = *(_QWORD *)(v56 + 40);
                goto LABEL_116;
              }
LABEL_113:
              v59 = *(_QWORD *)(v57 + 64);
              v60 = v57 + 48;
              if ( v59 != v60 )
              {
                v54 = 0;
                do
                {
                  v54 += sub_EF9210(v59 + 48);
                  v59 = sub_220EF30(v59);
                }
                while ( v60 != v59 );
LABEL_116:
                if ( v54 )
                {
LABEL_117:
                  v47 += v54;
                  v53 = (_QWORD *)sub_220EF30(v53);
                  if ( v67 == v53 )
                  {
                    v9 = v69;
                    v46 = v62;
                    v40 = v61;
                    goto LABEL_119;
                  }
                  continue;
                }
              }
            }
            else if ( v55 )
            {
              v57 = v53[24];
              goto LABEL_113;
            }
            break;
          }
          v54 = v53[13] != 0;
          goto LABEL_117;
        }
LABEL_47:
        v10 = v8[13] != 0;
LABEL_41:
        v83 += v10;
        v8 = (_QWORD *)sub_220EF30(v8);
        if ( v82 == v8 )
        {
          v2 = v83;
          v1 = a1;
          goto LABEL_45;
        }
      }
    }
  }
  v2 = *(_QWORD *)(v5 + 40);
LABEL_45:
  if ( !v2 )
    return v1[7] != 0;
  return v2;
}
