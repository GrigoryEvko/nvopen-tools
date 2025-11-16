// Function: sub_10088F0
// Address: 0x10088f0
//
__int64 *__fastcall sub_10088F0(__int64 *a1, __int64 *a2, char a3, __m128i *a4, char a5, char a6)
{
  __int64 *result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // r12d
  __int64 *v14; // r14
  void *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  bool v18; // al
  __int64 *v19; // r15
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 *v22; // rbx
  __int64 *v23; // rdi
  __int64 *v24; // rsi
  __int64 *v25; // rax
  int v26; // r14d
  unsigned int v27; // r15d
  void **v28; // rax
  void **v29; // rsi
  char v30; // al
  _BYTE *v31; // rsi
  __int64 *v32; // rsi
  __int64 v33; // r15
  void **v34; // rax
  unsigned __int8 *v35; // rdi
  void **v36; // r14
  void **v37; // r14
  void *v38; // rax
  _BYTE *v39; // rdx
  bool v40; // al
  __int64 v41; // r15
  _BYTE *v42; // rax
  void *v43; // rax
  _BYTE *v44; // r14
  __int64 v45; // r15
  void **v46; // rax
  void **v47; // r14
  void **v48; // r14
  bool v49; // al
  int v50; // r14d
  bool v51; // r15
  unsigned int v52; // ebx
  char *v53; // rax
  char v54; // al
  unsigned int v55; // r15d
  void **v56; // rax
  void **v57; // rsi
  char v58; // al
  _BYTE *v59; // rsi
  void **v60; // rax
  void **v61; // r14
  void **v62; // r14
  unsigned int v63; // r14d
  void **v64; // rax
  void **v65; // rsi
  char v66; // al
  void *v67; // rax
  _BYTE *v68; // rsi
  int v69; // [rsp+0h] [rbp-80h]
  unsigned __int8 v70; // [rsp+0h] [rbp-80h]
  __int64 *v71; // [rsp+8h] [rbp-78h]
  bool v72; // [rsp+8h] [rbp-78h]
  char v73; // [rsp+8h] [rbp-78h]
  __int64 v74; // [rsp+8h] [rbp-78h]
  __int64 v75; // [rsp+8h] [rbp-78h]
  __int64 *v76; // [rsp+10h] [rbp-70h]
  _BYTE *v77; // [rsp+10h] [rbp-70h]
  __int64 v78; // [rsp+10h] [rbp-70h]
  __int64 v79; // [rsp+10h] [rbp-70h]
  __int64 v80; // [rsp+10h] [rbp-70h]
  int v81; // [rsp+10h] [rbp-70h]
  bool v82; // [rsp+1Bh] [rbp-65h]
  __int64 *v84; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v85; // [rsp+28h] [rbp-58h] BYREF
  __int64 v86; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v87; // [rsp+38h] [rbp-48h] BYREF
  __int64 *v88; // [rsp+40h] [rbp-40h] BYREF
  __int64 *v89; // [rsp+48h] [rbp-38h]

  v85 = a1;
  v84 = a2;
  v82 = a6 == 1 && a5 == 0;
  if ( v82 )
  {
    result = (__int64 *)sub_FFE3E0(0x10u, (_BYTE **)&v85, (_BYTE **)&v84, a4->m128i_i64);
    if ( result )
      return result;
    v88 = v85;
    v89 = v84;
    result = (__int64 *)sub_1003820((__int64 *)&v88, 2, a3, (__int64)a4, 0, 1);
    if ( result )
      return result;
    v88 = 0;
    if ( (unsigned __int8)sub_10069D0(&v88, (__int64)v84) )
      return v85;
  }
  else
  {
    v88 = a1;
    v89 = a2;
    result = (__int64 *)sub_1003820((__int64 *)&v88, 2, a3, (__int64)a4, a5, a6);
    if ( result )
      return result;
    if ( a5 && (a3 & 2) == 0 )
      return 0;
    if ( (a6 & 0xFB) != 3 || (a3 & 8) != 0 )
    {
      v88 = 0;
      if ( (unsigned __int8)sub_10069D0(&v88, (__int64)v84) )
        return v85;
    }
    if ( a5 )
    {
      if ( (a3 & 2) == 0 )
        return 0;
      v88 = 0;
      if ( !(unsigned __int8)sub_1008640(&v88, (__int64)v84) )
        goto LABEL_13;
LABEL_28:
      if ( (a3 & 8) == 0 && (sub_9B4030(v85, 32, 0, a4) & 0x20) != 0 )
      {
        if ( !a5 )
          goto LABEL_31;
LABEL_13:
        if ( (a3 & 2) == 0 )
          return 0;
        v88 = 0;
        if ( (unsigned __int8)sub_1008640(&v88, (__int64)v85) )
        {
          v87 = &v86;
          if ( (unsigned __int8)sub_995E90(&v87, (unsigned __int64)v84, v10, v11, v12) )
            return (__int64 *)v86;
        }
        if ( (a3 & 2) == 0 )
          return 0;
        v13 = a3 & 8;
        if ( (a3 & 8) == 0 )
          return 0;
LABEL_17:
        v14 = v85;
        if ( *(_BYTE *)v85 == 18 )
        {
          v15 = sub_C33340();
          v10 = (__int64)(v14 + 3);
          if ( (void *)v14[3] == v15 )
            v10 = v14[4];
          v18 = (*(_BYTE *)(v10 + 20) & 7) == 3;
        }
        else
        {
          v41 = v85[1];
          if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 > 1 || *(_BYTE *)v85 > 0x15u )
            goto LABEL_33;
          v42 = sub_AD7630((__int64)v85, 0, v10);
          if ( !v42 || (v77 = v42, *v42 != 18) )
          {
            if ( *(_BYTE *)(v41 + 8) == 17 )
            {
              v69 = *(_DWORD *)(v41 + 32);
              if ( v69 )
              {
                v73 = 0;
                v55 = 0;
                while ( 1 )
                {
                  v56 = (void **)sub_AD69F0((unsigned __int8 *)v14, v55);
                  v57 = v56;
                  if ( !v56 )
                    break;
                  v58 = *(_BYTE *)v56;
                  if ( v58 != 13 )
                  {
                    if ( v58 != 18 )
                      goto LABEL_33;
                    v59 = v57[3] == sub_C33340() ? v57[4] : v57 + 3;
                    if ( (v59[20] & 7) != 3 )
                      goto LABEL_33;
                    v73 = 1;
                  }
                  if ( v69 == ++v55 )
                  {
                    if ( v73 )
                      goto LABEL_22;
                    goto LABEL_33;
                  }
                }
              }
            }
            goto LABEL_33;
          }
          v43 = sub_C33340();
          v10 = (__int64)v77;
          v44 = v77 + 24;
          if ( *((void **)v77 + 3) == v43 )
            v44 = (_BYTE *)*((_QWORD *)v77 + 4);
          v18 = (v44[20] & 7) == 3;
        }
        if ( !v18 )
          goto LABEL_33;
LABEL_22:
        v19 = v84;
        if ( *(_BYTE *)v84 != 45 )
          goto LABEL_23;
        v10 = *(v84 - 8);
        if ( *(_BYTE *)v10 == 18 )
        {
          v78 = *(v84 - 8);
          if ( *(void **)(v78 + 24) == sub_C33340() )
            v10 = *(_QWORD *)(v78 + 32);
          else
            v10 = v78 + 24;
          v49 = (*(_BYTE *)(v10 + 20) & 7) == 3;
        }
        else
        {
          v17 = *(_QWORD *)(v10 + 8);
          v80 = v17;
          if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 > 1 || *(_BYTE *)v10 > 0x15u )
          {
LABEL_23:
            v88 = &v86;
            if ( (unsigned __int8)sub_995E90(&v88, (unsigned __int64)v19, v10, v16, v17) )
              return (__int64 *)v86;
            goto LABEL_33;
          }
          v74 = *(v84 - 8);
          v60 = (void **)sub_AD7630(v10, 0, v10);
          v10 = v74;
          v17 = v80;
          v61 = v60;
          if ( !v60 || *(_BYTE *)v60 != 18 )
          {
            if ( *(_BYTE *)(v80 + 8) == 17 )
            {
              v63 = 0;
              v16 = 0;
              v81 = *(_DWORD *)(v80 + 32);
              while ( v81 != v63 )
              {
                v70 = v16;
                v75 = v10;
                v64 = (void **)sub_AD69F0((unsigned __int8 *)v10, v63);
                v10 = v75;
                v16 = v70;
                v65 = v64;
                if ( !v64 )
                  goto LABEL_108;
                v66 = *(_BYTE *)v64;
                if ( v66 != 13 )
                {
                  if ( v66 != 18 )
                    goto LABEL_108;
                  v67 = sub_C33340();
                  v10 = v75;
                  v68 = v65[3] == v67 ? v65[4] : v65 + 3;
                  if ( (v68[20] & 7) != 3 )
                    goto LABEL_108;
                  v16 = 1;
                }
                ++v63;
              }
              if ( (_BYTE)v16 )
              {
LABEL_107:
                result = (__int64 *)*(v19 - 4);
                if ( result )
                  return result;
              }
            }
LABEL_108:
            v19 = v84;
            goto LABEL_23;
          }
          if ( v60[3] == sub_C33340() )
            v62 = (void **)v61[4];
          else
            v62 = v61 + 3;
          v49 = (*((_BYTE *)v62 + 20) & 7) == 3;
        }
        if ( v49 )
          goto LABEL_107;
        goto LABEL_108;
      }
      return v85;
    }
  }
  v88 = 0;
  if ( (unsigned __int8)sub_1008640(&v88, (__int64)v84) )
    goto LABEL_28;
LABEL_31:
  v88 = 0;
  if ( (unsigned __int8)sub_1008640(&v88, (__int64)v85) )
  {
    v87 = &v86;
    if ( (unsigned __int8)sub_995E90(&v87, (unsigned __int64)v84, v10, v20, v21) )
      return (__int64 *)v86;
  }
  v13 = a3 & 8;
  if ( (a3 & 8) != 0 )
    goto LABEL_17;
LABEL_33:
  if ( !v82 )
    return 0;
  if ( (a3 & 2) == 0 )
    goto LABEL_53;
  v22 = v84;
  if ( v85 == v84 )
    return (__int64 *)sub_AD6530(v85[1], (__int64)v84);
  if ( *(_BYTE *)v85 == 18 )
  {
    v71 = v85;
    v23 = v85;
    v24 = v84;
    if ( (void *)v23[3] == sub_C33340() )
      v25 = (__int64 *)v71[4];
    else
      v25 = v71 + 3;
    if ( (*((_BYTE *)v25 + 20) & 7) == 0 )
      return v23;
  }
  else
  {
    v33 = v85[1];
    if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 > 1 || *(_BYTE *)v85 > 0x15u )
      goto LABEL_72;
    v76 = v85;
    v34 = (void **)sub_AD7630((__int64)v85, 0, v10);
    v35 = (unsigned __int8 *)v76;
    v36 = v34;
    if ( !v34 || *(_BYTE *)v34 != 18 )
    {
      if ( *(_BYTE *)(v33 + 8) == 17 )
      {
        v50 = *(_DWORD *)(v33 + 32);
        if ( v50 )
        {
          v51 = 0;
          v52 = 0;
          while ( 1 )
          {
            v53 = (char *)sub_AD69F0(v35, v52);
            v10 = (__int64)v53;
            if ( !v53 )
              break;
            v54 = *v53;
            v79 = v10;
            if ( v54 != 13 )
            {
              if ( v54 != 18 )
                break;
              v10 = *(void **)(v10 + 24) == sub_C33340() ? *(_QWORD *)(v79 + 32) : v79 + 24;
              if ( (*(_BYTE *)(v10 + 20) & 7) != 0 )
                break;
              v51 = v82;
            }
            if ( v50 == ++v52 )
            {
              if ( !v51 )
                break;
              return v85;
            }
          }
        }
      }
      v22 = v84;
      goto LABEL_72;
    }
    if ( v34[3] == sub_C33340() )
      v37 = (void **)v36[4];
    else
      v37 = v36 + 3;
    v23 = v85;
    v24 = v84;
    if ( (*((_BYTE *)v37 + 20) & 7) == 0 )
      return v23;
  }
  v22 = v24;
LABEL_72:
  if ( *(_BYTE *)v22 == 18 )
  {
    v38 = sub_C33340();
    v39 = v22 + 3;
    if ( (void *)v22[3] == v38 )
      v39 = (_BYTE *)v22[4];
    v40 = (v39[20] & 7) == 0;
LABEL_76:
    if ( v40 )
    {
LABEL_77:
      if ( *(_BYTE *)v84 <= 0x15u )
        return (__int64 *)sub_96E680(12, (__int64)v84);
      return 0;
    }
    goto LABEL_53;
  }
  v45 = v22[1];
  if ( (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17 <= 1 && *(_BYTE *)v22 <= 0x15u )
  {
    v46 = (void **)sub_AD7630((__int64)v22, 0, v10);
    v47 = v46;
    if ( v46 && *(_BYTE *)v46 == 18 )
    {
      if ( v46[3] == sub_C33340() )
        v48 = (void **)v47[4];
      else
        v48 = v47 + 3;
      v40 = (*((_BYTE *)v48 + 20) & 7) == 0;
      goto LABEL_76;
    }
    if ( *(_BYTE *)(v45 + 8) == 17 )
    {
      v26 = *(_DWORD *)(v45 + 32);
      if ( v26 )
      {
        v72 = 0;
        v27 = 0;
        while ( 1 )
        {
          v28 = (void **)sub_AD69F0((unsigned __int8 *)v22, v27);
          v29 = v28;
          if ( !v28 )
            break;
          v30 = *(_BYTE *)v28;
          if ( v30 != 13 )
          {
            if ( v30 != 18 )
              break;
            v31 = v29[3] == sub_C33340() ? v29[4] : v29 + 3;
            if ( (v31[20] & 7) != 0 )
              break;
            v72 = v82;
          }
          if ( v26 == ++v27 )
          {
            if ( v72 )
              goto LABEL_77;
            break;
          }
        }
      }
    }
  }
LABEL_53:
  if ( !v13 || (a3 & 1) == 0 )
    return 0;
  if ( *(_BYTE *)v84 != 45 || v85 != (__int64 *)*(v84 - 8) || (result = (__int64 *)*(v84 - 4)) == 0 )
  {
    if ( *(_BYTE *)v85 != 43 )
      return 0;
    v32 = (__int64 *)*(v85 - 8);
    result = (__int64 *)*(v85 - 4);
    if ( v84 == v32 && v32 )
    {
      if ( !result )
        return 0;
    }
    else
    {
      if ( result == 0 || v84 != result )
        return 0;
      result = (__int64 *)*(v85 - 8);
      if ( !v32 )
        return 0;
    }
  }
  return result;
}
