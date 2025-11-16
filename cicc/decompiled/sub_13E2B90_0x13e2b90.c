// Function: sub_13E2B90
// Address: 0x13e2b90
//
unsigned __int64 __fastcall sub_13E2B90(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 *a4)
{
  unsigned __int64 v4; // r10
  unsigned __int8 v8; // al
  unsigned __int64 result; // rax
  char v10; // al
  int v11; // eax
  int v12; // edx
  int v13; // edi
  __int64 v14; // rax
  char v15; // cl
  __int64 v16; // r9
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r8
  int v20; // ecx
  int v21; // eax
  __int64 **v22; // rax
  __int64 **v23; // rax
  __int64 **v24; // rax
  __int64 **v25; // rax
  __int64 v26; // r11
  int v27; // eax
  char v28; // al
  unsigned __int8 *v29; // r11
  unsigned __int64 v30; // rsi
  char v31; // al
  char v32; // al
  __int64 v33; // rax
  _BYTE *v34; // rdi
  __int64 **v35; // rax
  unsigned __int8 *v36; // r11
  unsigned __int8 *v37; // rdx
  __int64 **v38; // rax
  __int64 v39; // rax
  unsigned int v40; // r12d
  int v41; // eax
  bool v42; // al
  char v43; // al
  __int64 v44; // rax
  unsigned int v45; // r13d
  __int64 v46; // rax
  char v47; // dl
  unsigned int v48; // r12d
  int v49; // eax
  bool v50; // al
  __int64 v51; // r8
  __int64 v52; // rsi
  unsigned __int64 v53; // [rsp+8h] [rbp-78h]
  unsigned __int64 v54; // [rsp+8h] [rbp-78h]
  unsigned __int64 v55; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v56; // [rsp+10h] [rbp-70h]
  unsigned __int64 v57; // [rsp+10h] [rbp-70h]
  unsigned __int64 v58; // [rsp+10h] [rbp-70h]
  __int64 v59; // [rsp+10h] [rbp-70h]
  unsigned __int64 v60; // [rsp+10h] [rbp-70h]
  unsigned __int64 v61; // [rsp+10h] [rbp-70h]
  __int64 v62; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v63; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v64; // [rsp+18h] [rbp-68h]
  unsigned __int64 v65; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v66; // [rsp+18h] [rbp-68h]
  __int64 v67; // [rsp+18h] [rbp-68h]
  unsigned __int64 v68; // [rsp+18h] [rbp-68h]
  __int64 **v69; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v70; // [rsp+18h] [rbp-68h]
  __int64 v71; // [rsp+18h] [rbp-68h]
  __int64 v72; // [rsp+18h] [rbp-68h]
  int v73; // [rsp+18h] [rbp-68h]
  __int64 v74; // [rsp+20h] [rbp-60h]
  __int64 **v76; // [rsp+28h] [rbp-58h]
  int v77; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v78; // [rsp+28h] [rbp-58h]
  __int64 **v79; // [rsp+28h] [rbp-58h]
  unsigned __int64 v80; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v81; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 *v82; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 *v83; // [rsp+48h] [rbp-38h] BYREF

  v4 = a3;
  v8 = *(_BYTE *)(a1 + 16);
  if ( v8 <= 0x10u )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      if ( v8 == 9 )
      {
        if ( *(_BYTE *)(a3 + 16) <= 0x10u )
          return v4;
        return a2;
      }
    }
    else
    {
      if ( *(_BYTE *)(a3 + 16) <= 0x10u )
        return sub_1584040();
      if ( v8 == 9 )
        return a2;
    }
    if ( (unsigned __int8)sub_1596070(a1) )
      return a2;
    v10 = sub_1593BB0(a1);
    v4 = a3;
    if ( v10 )
      return v4;
  }
  if ( a2 == v4 || *(_BYTE *)(a2 + 16) == 9 )
    return v4;
  if ( *(_BYTE *)(v4 + 16) == 9 )
    return a2;
  v11 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v11 <= 0x17u )
    return 0;
  if ( (_BYTE)v11 == 75 )
  {
    v26 = *(_QWORD *)(a1 - 48);
    if ( v26 )
    {
      v74 = *(_QWORD *)(a1 - 24);
      if ( v74 )
      {
        v27 = *(unsigned __int16 *)(a1 + 18);
        BYTE1(v27) &= ~0x80u;
        v77 = v27;
        if ( (unsigned int)(v27 - 32) > 1 || *(_BYTE *)(v74 + 16) > 0x10u )
          goto LABEL_50;
        v58 = v4;
        v67 = *(_QWORD *)(a1 - 48);
        v31 = sub_1593BB0(v74);
        v26 = v67;
        v4 = v58;
        if ( v31 )
        {
LABEL_63:
          v82 = &v80;
          v83 = &v81;
          v32 = *(_BYTE *)(v26 + 16);
          if ( v32 == 50 )
          {
            if ( !*(_QWORD *)(v26 - 48) )
              goto LABEL_50;
            v80 = *(_QWORD *)(v26 - 48);
            v60 = v4;
            v71 = v26;
            v43 = sub_13D2630(&v83, *(_BYTE **)(v26 - 24));
            v26 = v71;
            v4 = v60;
            if ( !v43 )
              goto LABEL_50;
          }
          else
          {
            if ( v32 != 5 )
              goto LABEL_50;
            if ( *(_WORD *)(v26 + 18) != 26 )
              goto LABEL_50;
            v33 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
            if ( !*(_QWORD *)(v26 - 24 * v33) )
              goto LABEL_50;
            v80 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
            v34 = *(_BYTE **)(v26 + 24 * (1 - v33));
            if ( v34[16] == 13 )
            {
              v81 = (unsigned __int64)(v34 + 24);
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) != 16 )
                goto LABEL_50;
              v61 = v4;
              v72 = v26;
              v44 = sub_15A1020(v34);
              v26 = v72;
              v4 = v61;
              if ( !v44 || *(_BYTE *)(v44 + 16) != 13 )
                goto LABEL_50;
              *v83 = v44 + 24;
            }
          }
          v59 = v26;
          v68 = v4;
          result = sub_13D41B0(a2, v4, v80, v81, v77 == 32);
          v4 = v68;
          v26 = v59;
          if ( result )
            return result;
LABEL_50:
          v55 = v4;
          LODWORD(v80) = v77;
          v64 = (unsigned __int8 *)v26;
          LODWORD(v83) = 1;
          v82 = 0;
          v28 = sub_14CF800(v26, v74, &v80, &v81, &v82, 1);
          v29 = v64;
          v4 = v55;
          if ( v28 )
          {
            v30 = v55;
            v56 = v64;
            v65 = v4;
            result = sub_13D41B0(a2, v30, v81, (__int64)&v82, (_DWORD)v80 == 32);
            v4 = v65;
            v29 = v56;
            if ( (unsigned int)v83 <= 0x40 )
              goto LABEL_54;
          }
          else
          {
            result = 0;
            if ( (unsigned int)v83 <= 0x40 )
            {
LABEL_55:
              if ( v77 != 32 )
              {
                if ( v77 == 33 )
                {
                  v69 = (__int64 **)v4;
                  v78 = v29;
                  v35 = sub_13E2680((_QWORD *)a2, v29, (unsigned __int8 *)v74, a4);
                  v36 = v78;
                  if ( v69 == v35 )
                    return a2;
                  v37 = v78;
                  v79 = v69;
                  v70 = v36;
                  if ( v79 == sub_13E2680((_QWORD *)a2, (unsigned __int8 *)v74, v37, a4) )
                    return a2;
                  if ( (__int64 **)a2 == sub_13E2680(v79, v70, (unsigned __int8 *)v74, a4) )
                    return a2;
                  v38 = sub_13E2680(v79, (unsigned __int8 *)v74, v70, a4);
                  v4 = (unsigned __int64)v79;
                  if ( (__int64 **)a2 == v38 )
                    return a2;
                }
                goto LABEL_57;
              }
              v63 = v29;
              v76 = (__int64 **)v4;
              v22 = sub_13E2680((_QWORD *)v4, v29, (unsigned __int8 *)v74, a4);
              v4 = (unsigned __int64)v76;
              if ( (__int64 **)a2 != v22 )
              {
                v23 = sub_13E2680(v76, (unsigned __int8 *)v74, v63, a4);
                v4 = (unsigned __int64)v76;
                if ( (__int64 **)a2 != v23 )
                {
                  v24 = sub_13E2680((_QWORD *)a2, v63, (unsigned __int8 *)v74, a4);
                  v4 = (unsigned __int64)v76;
                  if ( v76 != v24 )
                  {
                    v25 = sub_13E2680((_QWORD *)a2, (unsigned __int8 *)v74, v63, a4);
                    v4 = (unsigned __int64)v76;
                    if ( v76 != v25 )
                    {
LABEL_57:
                      v11 = *(unsigned __int8 *)(a1 + 16);
                      if ( (unsigned __int8)v11 <= 0x17u )
                        return 0;
                      goto LABEL_13;
                    }
                  }
                }
              }
              return v4;
            }
          }
          if ( v82 )
          {
            v53 = v4;
            v57 = result;
            v66 = v29;
            j_j___libc_free_0_0(v82);
            v4 = v53;
            result = v57;
            v29 = v66;
          }
LABEL_54:
          if ( result )
            return result;
          goto LABEL_55;
        }
        v39 = v74;
        if ( *(_BYTE *)(v74 + 16) == 13 )
        {
          v40 = *(_DWORD *)(v74 + 32);
          if ( v40 > 0x40 )
          {
            v39 = v74;
LABEL_79:
            v41 = sub_16A57B0(v39 + 24);
            v26 = v67;
            v4 = v58;
            v42 = v40 == v41;
            goto LABEL_80;
          }
        }
        else
        {
          if ( *(_BYTE *)(*(_QWORD *)v74 + 8LL) != 16 )
            goto LABEL_50;
          v39 = sub_15A1020(v74);
          v26 = v67;
          v4 = v58;
          if ( !v39 || *(_BYTE *)(v39 + 16) != 13 )
          {
            v45 = 0;
            v73 = *(_DWORD *)(*(_QWORD *)v74 + 32LL);
            while ( v73 != v45 )
            {
              v54 = v4;
              v62 = v26;
              v46 = sub_15A0A60(v74, v45);
              v26 = v62;
              v4 = v54;
              if ( !v46 )
                goto LABEL_50;
              v47 = *(_BYTE *)(v46 + 16);
              if ( v47 != 9 )
              {
                if ( v47 != 13 )
                  goto LABEL_50;
                v48 = *(_DWORD *)(v46 + 32);
                if ( v48 <= 0x40 )
                {
                  v50 = *(_QWORD *)(v46 + 24) == 0;
                }
                else
                {
                  v49 = sub_16A57B0(v46 + 24);
                  v26 = v62;
                  v4 = v54;
                  v50 = v48 == v49;
                }
                if ( !v50 )
                  goto LABEL_50;
              }
              ++v45;
            }
            goto LABEL_63;
          }
          v40 = *(_DWORD *)(v39 + 32);
          if ( v40 > 0x40 )
            goto LABEL_79;
        }
        v42 = *(_QWORD *)(v39 + 24) == 0;
LABEL_80:
        if ( !v42 )
          goto LABEL_50;
        goto LABEL_63;
      }
    }
  }
LABEL_13:
  if ( (unsigned int)(v11 - 35) > 0x11 )
    return 0;
  v12 = v11 - 24;
  if ( v11 == 51 )
  {
    v13 = 33;
  }
  else
  {
    v13 = 32;
    if ( v11 != 50 )
      return 0;
  }
  v14 = *(_QWORD *)(a1 - 48);
  v15 = *(_BYTE *)(v14 + 16);
  if ( v15 != 75 )
    goto LABEL_17;
  v51 = *(_QWORD *)(v14 - 48);
  if ( !v51 )
    goto LABEL_17;
  v52 = *(_QWORD *)(v14 - 24);
  if ( a2 != v51 )
  {
    if ( a2 == v52 && v52 != 0 && v4 == v51 )
      goto LABEL_109;
LABEL_17:
    v16 = *(_QWORD *)(a1 - 24);
    if ( *(_BYTE *)(v16 + 16) != 75 )
      return 0;
    v17 = *(_QWORD *)(v16 - 48);
    if ( !v17 )
      return 0;
    goto LABEL_19;
  }
  if ( !v52 || v4 != v52 )
    goto LABEL_17;
LABEL_109:
  v16 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v16 + 16) != 75 )
    return 0;
  v17 = *(_QWORD *)(v16 - 48);
  if ( !v17 )
    return 0;
  v19 = *(_QWORD *)(v16 - 24);
  if ( v19 )
  {
    v20 = *(unsigned __int16 *)(v14 + 18);
    v21 = *(unsigned __int16 *)(v16 + 18);
    BYTE1(v20) &= ~0x80u;
    BYTE1(v21) &= ~0x80u;
    goto LABEL_26;
  }
LABEL_19:
  v18 = *(_QWORD *)(v16 - 24);
  if ( a2 == v17 )
  {
    if ( v4 != v18 || !v18 )
      return 0;
  }
  else if ( v18 == 0 || a2 != v18 || v4 != v17 )
  {
    return 0;
  }
  if ( v15 != 75 )
    return 0;
  v17 = *(_QWORD *)(v14 - 48);
  if ( !v17 )
    return 0;
  v19 = *(_QWORD *)(v14 - 24);
  if ( !v19 )
    return 0;
  v20 = *(unsigned __int16 *)(v16 + 18);
  v21 = *(unsigned __int16 *)(v14 + 18);
  BYTE1(v20) &= ~0x80u;
  BYTE1(v21) &= ~0x80u;
LABEL_26:
  if ( v21 != v20 || v13 != v21 || a2 != v17 && v4 != v17 && v4 != v19 && a2 != v19 )
    return 0;
  result = v4;
  if ( v12 == 27 )
    return a2;
  return result;
}
