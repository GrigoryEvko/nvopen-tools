// Function: sub_14616B0
// Address: 0x14616b0
//
__int64 __fastcall sub_14616B0(_QWORD *a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, unsigned int a5)
{
  unsigned __int64 v6; // r12
  _QWORD *v7; // rcx
  _QWORD *v10; // r9
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rdx
  _BYTE *v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rcx
  _BYTE *v20; // rcx
  __int64 v21; // rsi
  unsigned __int8 v22; // al
  unsigned __int8 v23; // dl
  int v24; // eax
  __int64 v25; // rdx
  __int64 result; // rax
  __int64 v27; // rsi
  __int64 v28; // rbx
  _QWORD **v29; // rax
  _QWORD **v30; // rsi
  _QWORD *v31; // rdx
  int v32; // eax
  int v33; // ecx
  _QWORD *v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r15
  _QWORD *v40; // rax
  _QWORD *v41; // r9
  _QWORD *v42; // r12
  unsigned __int64 v43; // rbx
  _QWORD *v44; // rcx
  _QWORD *v45; // r13
  _QWORD *v46; // r14
  _QWORD *v47; // rdx
  _BYTE *v48; // rdi
  _BYTE *v49; // rax
  _BYTE *v50; // r12
  __int64 v51; // rcx
  _BYTE *v52; // r13
  _BYTE *v53; // r14
  _QWORD *v54; // rdx
  _BYTE *v55; // rdi
  _BYTE *v56; // rax
  _BYTE *v57; // rsi
  _BYTE *v58; // rcx
  _QWORD *v59; // rax
  _BYTE **v60; // r11
  _BYTE *v61; // rdi
  _BYTE *v62; // rax
  _QWORD **v63; // r11
  _BYTE *v64; // rax
  _BYTE *v65; // rdi
  _BYTE *v66; // rax
  const void *v67; // r14
  size_t v68; // rdx
  size_t v69; // r12
  size_t v70; // rdx
  const void *v71; // rsi
  size_t v72; // rbx
  int v73; // eax
  _QWORD *v74; // [rsp-98h] [rbp-98h]
  unsigned int v75; // [rsp-98h] [rbp-98h]
  _BYTE *v76; // [rsp-90h] [rbp-90h]
  unsigned int v77; // [rsp-88h] [rbp-88h]
  _QWORD **v78; // [rsp-88h] [rbp-88h]
  _BYTE *v79; // [rsp-80h] [rbp-80h]
  _QWORD *v80; // [rsp-78h] [rbp-78h]
  _BYTE *v81; // [rsp-78h] [rbp-78h]
  _BYTE *v82; // [rsp-78h] [rbp-78h]
  unsigned int v83; // [rsp-70h] [rbp-70h]
  __int64 v84; // [rsp-70h] [rbp-70h]
  _QWORD *v85; // [rsp-70h] [rbp-70h]
  _QWORD *v86; // [rsp-70h] [rbp-70h]
  _QWORD *v87; // [rsp-68h] [rbp-68h]
  unsigned int v88; // [rsp-68h] [rbp-68h]
  _QWORD *v89; // [rsp-68h] [rbp-68h]
  _QWORD *v90; // [rsp-68h] [rbp-68h]
  _QWORD *v91; // [rsp-68h] [rbp-68h]
  _QWORD *v92; // [rsp-68h] [rbp-68h]
  _QWORD *v93; // [rsp-68h] [rbp-68h]
  _QWORD *v94; // [rsp-60h] [rbp-60h]
  __int64 v95; // [rsp-60h] [rbp-60h]
  _QWORD *v96; // [rsp-60h] [rbp-60h]
  __int64 v97; // [rsp-60h] [rbp-60h]
  _QWORD *v98; // [rsp-60h] [rbp-60h]
  _QWORD *v99; // [rsp-60h] [rbp-60h]
  __int64 v100; // [rsp-58h] [rbp-58h] BYREF
  __int64 v101; // [rsp-50h] [rbp-50h]
  unsigned __int64 v102; // [rsp-48h] [rbp-48h]

  if ( a5 > dword_4F9AEE0 )
    return 0;
  v6 = a3;
  if ( a3 == a4 )
    return 0;
  v7 = (_QWORD *)a1[2];
  v102 = a3;
  v100 = (__int64)&v100;
  v10 = a1 + 1;
  v101 = 1;
  v11 = v7;
  if ( v7 )
  {
    v12 = a1 + 1;
    do
    {
      while ( 1 )
      {
        v13 = v11[2];
        v14 = v11[3];
        if ( v11[6] >= v6 )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v14 )
          goto LABEL_8;
      }
      v12 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v13 );
LABEL_8:
    if ( v10 != v12 && v12[6] <= v6 )
    {
      v15 = v12 + 4;
      if ( (v12[5] & 1) == 0 )
      {
        v15 = (_BYTE *)v12[4];
        if ( (v15[8] & 1) == 0 )
        {
          v57 = *(_BYTE **)v15;
          if ( (*(_BYTE *)(*(_QWORD *)v15 + 8LL) & 1) != 0 )
          {
            v12[4] = v57;
            v15 = v57;
            v7 = (_QWORD *)a1[2];
          }
          else
          {
            v58 = *(_BYTE **)v57;
            if ( (*(_BYTE *)(*(_QWORD *)v57 + 8LL) & 1) == 0 )
            {
              v59 = *(_QWORD **)v58;
              v98 = *(_QWORD **)v58;
              if ( (*(_BYTE *)(*(_QWORD *)v58 + 8LL) & 1) != 0 )
              {
                v58 = *(_BYTE **)v58;
              }
              else
              {
                v60 = (_BYTE **)*v59;
                if ( (*(_BYTE *)(*v59 + 8LL) & 1) == 0 )
                {
                  v61 = *v60;
                  v74 = (_QWORD *)*v59;
                  v60 = (_BYTE **)v61;
                  if ( (v61[8] & 1) == 0 )
                  {
                    v77 = a5;
                    v79 = (_BYTE *)v12[4];
                    v81 = *(_BYTE **)v57;
                    v92 = v10;
                    v62 = sub_145F3B0(v61);
                    v10 = v92;
                    v58 = v81;
                    v15 = v79;
                    a5 = v77;
                    *v74 = v62;
                    v60 = (_BYTE **)v62;
                  }
                  *v98 = v60;
                }
                *(_QWORD *)v58 = v60;
                v58 = v60;
              }
              *(_QWORD *)v57 = v58;
            }
            *(_QWORD *)v15 = v58;
            v12[4] = v58;
            if ( !v58 )
              goto LABEL_25;
            v15 = v58;
            v7 = (_QWORD *)a1[2];
          }
        }
      }
      v100 = (__int64)&v100;
      v16 = v7;
      v101 = 1;
      v102 = a4;
      if ( v7 )
      {
        v17 = v10;
        do
        {
          while ( 1 )
          {
            v18 = v16[2];
            v19 = v16[3];
            if ( v16[6] >= a4 )
              break;
            v16 = (_QWORD *)v16[3];
            if ( !v19 )
              goto LABEL_17;
          }
          v17 = v16;
          v16 = (_QWORD *)v16[2];
        }
        while ( v18 );
LABEL_17:
        if ( v10 != v17 && v17[6] <= a4 )
        {
          v20 = v17 + 4;
          if ( (v17[5] & 1) == 0 )
          {
            v20 = (_BYTE *)v17[4];
            if ( (v20[8] & 1) == 0 )
            {
              v21 = *(_QWORD *)v20;
              if ( (*(_BYTE *)(*(_QWORD *)v20 + 8LL) & 1) != 0 )
              {
                v20 = *(_BYTE **)v20;
              }
              else
              {
                v63 = *(_QWORD ***)v21;
                if ( (*(_BYTE *)(*(_QWORD *)v21 + 8LL) & 1) == 0 )
                {
                  v93 = *v63;
                  if ( ((*v63)[1] & 1) != 0 )
                  {
                    v63 = (_QWORD **)*v63;
                  }
                  else
                  {
                    v64 = (_BYTE *)**v63;
                    v99 = v64;
                    if ( (v64[8] & 1) == 0 )
                    {
                      v65 = *(_BYTE **)v64;
                      if ( (*(_BYTE *)(*(_QWORD *)v64 + 8LL) & 1) == 0 )
                      {
                        v75 = a5;
                        v76 = v15;
                        v78 = *(_QWORD ***)v21;
                        v82 = (_BYTE *)v17[4];
                        v86 = v10;
                        v66 = sub_145F3B0(v65);
                        a5 = v75;
                        v65 = v66;
                        v15 = v76;
                        v63 = v78;
                        v20 = v82;
                        v10 = v86;
                        *v99 = v66;
                      }
                      v99 = v65;
                      *v93 = v65;
                    }
                    *v63 = v99;
                    v63 = (_QWORD **)v99;
                  }
                  *(_QWORD *)v21 = v63;
                }
                *(_QWORD *)v20 = v63;
                v20 = v63;
              }
              v17[4] = v20;
            }
          }
          if ( v20 == v15 )
            return 0;
        }
      }
    }
  }
LABEL_25:
  v22 = *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 15;
  v23 = *(_BYTE *)(*(_QWORD *)a4 + 8LL) == 15;
  if ( v22 != v23 )
    return v22 - (unsigned int)v23;
  v24 = *(unsigned __int8 *)(v6 + 16);
  LODWORD(v25) = *(unsigned __int8 *)(a4 + 16);
  if ( (_BYTE)v24 != (_BYTE)v25 )
    return (unsigned int)(v24 - v25);
  if ( (_BYTE)v24 == 17 )
    return (unsigned int)(*(_DWORD *)(v6 + 32) - *(_DWORD *)(a4 + 32));
  if ( (unsigned __int8)v24 > 3u || (*(_BYTE *)(v6 + 32) & 0xFu) - 7 <= 1 || (*(_BYTE *)(a4 + 32) & 0xFu) - 7 <= 1 )
  {
    if ( (unsigned __int8)v24 <= 0x17u )
      goto LABEL_48;
    v27 = *(_QWORD *)(v6 + 40);
    v28 = *(_QWORD *)(a4 + 40);
    if ( v28 == v27 )
      goto LABEL_42;
    v83 = a5;
    v87 = v10;
    v94 = (_QWORD *)sub_13AE450(a2, v27);
    v29 = (_QWORD **)sub_13AE450(a2, v28);
    v10 = v87;
    a5 = v83;
    v30 = v29;
    if ( v94 )
    {
      v31 = (_QWORD *)*v94;
      if ( *v94 )
      {
        v32 = 1;
        do
        {
          v31 = (_QWORD *)*v31;
          ++v32;
        }
        while ( v31 );
        v33 = 0;
        if ( !v30 )
        {
LABEL_41:
          if ( v32 == v33 )
            goto LABEL_42;
          return (unsigned int)(v32 - v33);
        }
        v34 = *v30;
        if ( !*v30 )
        {
          v33 = 1;
          goto LABEL_41;
        }
LABEL_39:
        v33 = 1;
        do
        {
          v34 = (_QWORD *)*v34;
          ++v33;
        }
        while ( v34 );
        goto LABEL_41;
      }
      if ( !v29 )
        return 1;
      v34 = *v29;
      if ( *v29 )
      {
        v32 = 1;
        goto LABEL_39;
      }
    }
    else if ( v29 )
    {
      v34 = *v29;
      if ( !*v29 )
      {
        v33 = 1;
        v32 = 0;
        return (unsigned int)(v32 - v33);
      }
      v32 = 0;
      goto LABEL_39;
    }
LABEL_42:
    v24 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
    v25 = *(_DWORD *)(a4 + 20) & 0xFFFFFFF;
    if ( (_DWORD)v25 == v24 )
    {
      if ( (_DWORD)v25 )
      {
        v80 = v10;
        v35 = 0;
        v88 = a5 + 1;
        v95 = v6;
        v84 = 24 * v25;
        while ( 1 )
        {
          v36 = sub_13CF970(a4);
          v37 = sub_13CF970(v95);
          result = sub_14616B0(a1, a2, *(_QWORD *)(v37 + v35), *(_QWORD *)(v36 + v35), v88);
          if ( (_DWORD)result )
            return result;
          v35 += 24;
          if ( v84 == v35 )
          {
            v10 = v80;
            v6 = v95;
            break;
          }
        }
      }
LABEL_48:
      v89 = v10;
      v100 = (__int64)&v100;
      v102 = v6;
      v101 = 1;
      v38 = sub_1461570(a1, (__int64)&v100);
      v102 = a4;
      v101 = 1;
      v39 = v38;
      v100 = (__int64)&v100;
      v40 = sub_1461570(a1, (__int64)&v100);
      v41 = v89;
      v42 = v40;
      if ( v89 == v40 )
      {
        v43 = 0;
        if ( v89 == v39 )
          return 0;
      }
      else
      {
        v43 = (unsigned __int64)(v40 + 4);
        if ( (v40[5] & 1) != 0 || (v43 = v40[4], (*(_BYTE *)(v43 + 8) & 1) != 0) )
        {
          if ( v89 == v39 )
          {
            v50 = 0;
            goto LABEL_76;
          }
        }
        else
        {
          v44 = *(_QWORD **)v43;
          if ( (*(_BYTE *)(*(_QWORD *)v43 + 8LL) & 1) != 0 )
          {
            v43 = *(_QWORD *)v43;
          }
          else
          {
            v45 = (_QWORD *)*v44;
            if ( (*(_BYTE *)(*v44 + 8LL) & 1) == 0 )
            {
              v46 = (_QWORD *)*v45;
              if ( (*(_BYTE *)(*v45 + 8LL) & 1) == 0 )
              {
                v47 = (_QWORD *)*v46;
                if ( (*(_BYTE *)(*v46 + 8LL) & 1) == 0 )
                {
                  v48 = (_BYTE *)*v47;
                  v85 = (_QWORD *)*v46;
                  if ( (*(_BYTE *)(*v47 + 8LL) & 1) == 0 )
                  {
                    v90 = *(_QWORD **)v43;
                    v96 = v41;
                    v49 = sub_145F3B0(v48);
                    v44 = v90;
                    v41 = v96;
                    v48 = v49;
                    *v85 = v49;
                  }
                  *v46 = v48;
                  v47 = v48;
                }
                *v45 = v47;
                v46 = v47;
              }
              *v44 = v46;
              v45 = v46;
            }
            *(_QWORD *)v43 = v45;
            v43 = (unsigned __int64)v45;
          }
          v42[4] = v43;
          v50 = 0;
          if ( v41 == v39 )
          {
LABEL_75:
            if ( v50 == (_BYTE *)v43 )
              return 0;
LABEL_76:
            *(_QWORD *)(*(_QWORD *)v50 + 8LL) = v43 | *(_QWORD *)(*(_QWORD *)v50 + 8LL) & 1LL;
            *(_QWORD *)v50 = *(_QWORD *)v43;
            *(_QWORD *)(v43 + 8) &= ~1uLL;
            *(_QWORD *)v43 = v50;
            return 0;
          }
        }
      }
      v50 = v39 + 4;
      if ( (v39[5] & 1) == 0 )
      {
        v50 = (_BYTE *)v39[4];
        if ( (v50[8] & 1) == 0 )
        {
          v51 = *(_QWORD *)v50;
          if ( (*(_BYTE *)(*(_QWORD *)v50 + 8LL) & 1) != 0 )
          {
            v50 = *(_BYTE **)v50;
          }
          else
          {
            v52 = *(_BYTE **)v51;
            if ( (*(_BYTE *)(*(_QWORD *)v51 + 8LL) & 1) == 0 )
            {
              v53 = *(_BYTE **)v52;
              if ( (*(_BYTE *)(*(_QWORD *)v52 + 8LL) & 1) == 0 )
              {
                v54 = *(_QWORD **)v53;
                if ( (*(_BYTE *)(*(_QWORD *)v53 + 8LL) & 1) == 0 )
                {
                  v55 = (_BYTE *)*v54;
                  v91 = *(_QWORD **)v53;
                  if ( (*(_BYTE *)(*v54 + 8LL) & 1) == 0 )
                  {
                    v97 = *(_QWORD *)v50;
                    v56 = sub_145F3B0(v55);
                    v51 = v97;
                    v55 = v56;
                    *v91 = v56;
                  }
                  *(_QWORD *)v53 = v55;
                  v54 = v55;
                }
                *(_QWORD *)v52 = v54;
                v53 = v54;
              }
              *(_QWORD *)v51 = v53;
              v52 = v53;
            }
            *(_QWORD *)v50 = v52;
            v50 = v52;
          }
          v39[4] = v50;
        }
      }
      goto LABEL_75;
    }
    return (unsigned int)(v24 - v25);
  }
  v67 = (const void *)sub_1649960(v6);
  v69 = v68;
  v71 = (const void *)sub_1649960(a4);
  v72 = v70;
  if ( v69 > v70 )
  {
    result = 1;
    if ( !v70 )
      return result;
    v73 = memcmp(v67, v71, v70);
    if ( !v73 )
      return v69 < v72 ? -1 : 1;
    return (v73 >> 31) | 1u;
  }
  if ( v69 )
  {
    v73 = memcmp(v67, v71, v69);
    if ( v73 )
      return (v73 >> 31) | 1u;
  }
  result = 0;
  if ( v69 != v72 )
    return v69 < v72 ? -1 : 1;
  return result;
}
