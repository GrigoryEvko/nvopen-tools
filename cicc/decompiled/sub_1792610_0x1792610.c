// Function: sub_1792610
// Address: 0x1792610
//
__int64 __fastcall sub_1792610(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  int v9; // eax
  int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rsi
  __int64 *v17; // r10
  unsigned int v18; // ecx
  __int64 v19; // r14
  const char *v20; // rax
  __int64 v21; // r11
  __int64 v22; // rdx
  const char *v23; // r10
  int v24; // edi
  int v25; // edx
  __int64 **v26; // rax
  __int64 v27; // rcx
  _QWORD **v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 *v34; // r13
  __int64 *v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  int v38; // edi
  int v39; // edi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // rsi
  int v47; // esi
  __int64 v48; // rsi
  __int64 v49; // rdi
  int v50; // eax
  __int64 v51; // r15
  bool v52; // al
  const char *v53; // r10
  bool v54; // al
  _QWORD *v55; // rax
  __int64 v56; // r12
  __int64 v57; // rax
  unsigned int v58; // r13d
  __int64 *v59; // rax
  __int64 *v60; // r10
  __int64 v61; // rax
  __int64 v62; // r14
  const char *v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 *v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // r12
  __int64 v70; // rax
  unsigned int v71; // r13d
  __int64 *v72; // rax
  __int64 *v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // [rsp+0h] [rbp-80h]
  __int64 v77; // [rsp+8h] [rbp-78h]
  const char *v78; // [rsp+8h] [rbp-78h]
  const char *v79; // [rsp+8h] [rbp-78h]
  __int64 *v80; // [rsp+18h] [rbp-68h]
  const char *v81; // [rsp+20h] [rbp-60h] BYREF
  __int64 v82; // [rsp+28h] [rbp-58h]
  const char **v83; // [rsp+30h] [rbp-50h] BYREF
  const char *v84; // [rsp+38h] [rbp-48h]
  __int16 v85; // [rsp+40h] [rbp-40h]

  v8 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v8 + 16) != 75 )
    goto LABEL_2;
  v41 = *(_QWORD *)(a2 - 48);
  v42 = *(_QWORD *)(a2 - 24);
  v43 = *(_QWORD *)(v8 - 48);
  v44 = *(_QWORD *)(v8 - 24);
  if ( v41 == v43 && v42 == v44 )
  {
    if ( (*(_WORD *)(v8 + 18) & 0x7FFFu) - 40 <= 1 )
      return 0;
    goto LABEL_54;
  }
  if ( v41 != v44 )
    goto LABEL_55;
  if ( v42 != v43 )
  {
LABEL_71:
    if ( v41 != v44 || v42 != v43 )
      goto LABEL_55;
LABEL_73:
    if ( v43 != v41 )
    {
      if ( (unsigned int)sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF) - 38 <= 1 )
        return 0;
      v8 = *(_QWORD *)(a2 - 72);
      if ( *(_BYTE *)(v8 + 16) != 75 )
        goto LABEL_2;
      v41 = *(_QWORD *)(a2 - 48);
      v42 = *(_QWORD *)(a2 - 24);
LABEL_55:
      v45 = *(_QWORD *)(v8 - 48);
      v46 = *(_QWORD *)(v8 - 24);
      if ( v41 == v45 && v42 == v46 )
      {
        v47 = *(unsigned __int16 *)(v8 + 18);
      }
      else
      {
        if ( v41 != v46 || v42 != v45 )
          goto LABEL_59;
        v47 = *(unsigned __int16 *)(v8 + 18);
        if ( v41 != v45 )
        {
          if ( (unsigned int)sub_15FF0F0(*(_WORD *)(v8 + 18) & 0x7FFF) - 36 <= 1 )
            return 0;
          v8 = *(_QWORD *)(a2 - 72);
          if ( *(_BYTE *)(v8 + 16) != 75 )
            goto LABEL_2;
          v41 = *(_QWORD *)(a2 - 48);
          v42 = *(_QWORD *)(a2 - 24);
LABEL_59:
          v48 = *(_QWORD *)(v8 - 48);
          v49 = *(_QWORD *)(v8 - 24);
          if ( v41 == v48 && v42 == v49 )
          {
            v50 = *(unsigned __int16 *)(v8 + 18);
          }
          else
          {
            if ( v41 != v49 || v42 != v48 )
              goto LABEL_2;
            v50 = *(unsigned __int16 *)(v8 + 18);
            if ( v41 != v48 )
            {
              v50 = sub_15FF0F0(v50 & 0xFFFF7FFF);
LABEL_63:
              if ( (unsigned int)(v50 - 34) <= 1 )
                return 0;
              goto LABEL_2;
            }
          }
          BYTE1(v50) &= ~0x80u;
          goto LABEL_63;
        }
      }
      if ( (v47 & 0xFFFF7FFF) - 36 <= 1 )
        return 0;
      goto LABEL_59;
    }
LABEL_54:
    if ( (*(_WORD *)(v8 + 18) & 0x7FFFu) - 38 <= 1 )
      return 0;
    goto LABEL_55;
  }
  v39 = *(_WORD *)(v8 + 18) & 0x7FFF;
  if ( v41 == v43 )
  {
    if ( (unsigned int)(v39 - 40) <= 1 )
      return 0;
    goto LABEL_73;
  }
  if ( (unsigned int)sub_15FF0F0(v39) - 40 <= 1 )
    return 0;
  v8 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v8 + 16) == 75 )
  {
    v41 = *(_QWORD *)(a2 - 48);
    v43 = *(_QWORD *)(v8 - 48);
    v42 = *(_QWORD *)(a2 - 24);
    v44 = *(_QWORD *)(v8 - 24);
    if ( v41 == v43 && v42 == v44 )
      goto LABEL_54;
    goto LABEL_71;
  }
LABEL_2:
  v9 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( v9 == 1 )
  {
    v25 = *(unsigned __int8 *)(a3 + 16);
    if ( (unsigned int)(v25 - 60) > 0xC )
      return 0;
    if ( (*(_BYTE *)(a4 + 23) & 0x40) != 0 )
      v26 = *(__int64 ***)(a4 - 8);
    else
      v26 = (__int64 **)(a4 - 24LL * (*(_DWORD *)(a4 + 20) & 0xFFFFFFF));
    v27 = **v26;
    v28 = (_QWORD **)(a3 - 24);
    if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
      v28 = *(_QWORD ***)(a3 - 8);
    if ( v27 != **v28 )
      return 0;
    v29 = **(_QWORD **)(a2 - 72);
    if ( *(_BYTE *)(v29 + 8) == 16 )
    {
      if ( *(_BYTE *)(v27 + 8) != 16 || *(_DWORD *)(v29 + 32) != *(_DWORD *)(v27 + 32) )
        return 0;
      if ( (_BYTE)v25 == 71 )
        goto LABEL_36;
    }
    v30 = *(_QWORD *)(a3 + 8);
    if ( v30 )
    {
      if ( !*(_QWORD *)(v30 + 8) )
      {
        v31 = *(_QWORD *)(a4 + 8);
        if ( v31 )
        {
          if ( !*(_QWORD *)(v31 + 8) )
          {
LABEL_36:
            v32 = *(_QWORD *)(a1 + 8);
            v81 = sub_1649960(a2);
            v83 = &v81;
            v82 = v33;
            v85 = 773;
            v84 = ".v";
            if ( (*(_BYTE *)(a4 + 23) & 0x40) != 0 )
              v34 = *(__int64 **)(a4 - 8);
            else
              v34 = (__int64 *)(a4 - 24LL * (*(_DWORD *)(a4 + 20) & 0xFFFFFFF));
            if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
              v35 = *(__int64 **)(a3 - 8);
            else
              v35 = (__int64 *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
            v36 = sub_1707C10(v32, *(_QWORD *)(a2 - 72), *v35, *v34, (__int64 *)&v83, a2);
            v37 = *(_QWORD *)a3;
            v38 = *(unsigned __int8 *)(a3 + 16);
            v85 = 257;
            return sub_15FDBD0(v38 - 24, (__int64)v36, v37, (__int64)&v83, 0);
          }
        }
      }
    }
    return 0;
  }
  if ( v9 != 2 )
    return 0;
  if ( (*(_DWORD *)(a4 + 20) & 0xFFFFFFF) != 2 )
    return 0;
  v10 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned int)(v10 - 35) > 0x11 && (_BYTE)v10 != 56 )
    return 0;
  v11 = *(_QWORD *)(a3 + 8);
  if ( !v11 )
    return 0;
  if ( *(_QWORD *)(v11 + 8) )
    return 0;
  v12 = *(_QWORD *)(a4 + 8);
  if ( !v12 )
    return 0;
  v13 = *(_QWORD *)(v12 + 8);
  if ( v13 )
    return 0;
  v14 = a3 - 48;
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    v14 = *(_QWORD *)(a3 - 8);
  v80 = *(__int64 **)v14;
  v15 = (__int64 *)(a4 - 48);
  if ( (*(_BYTE *)(a4 + 23) & 0x40) != 0 )
    v15 = *(__int64 **)(a4 - 8);
  v16 = v15[3];
  v17 = *(__int64 **)(v14 + 24);
  if ( (__int64 *)*v15 == v80 )
    goto LABEL_22;
  if ( v17 != (__int64 *)v16 )
  {
    v18 = v10 - 24;
    if ( v18 > 0x1C || ((1LL << v18) & 0x1C019800) == 0 )
      return v13;
    if ( v80 == (__int64 *)v16 )
    {
      v16 = *v15;
      goto LABEL_22;
    }
    if ( v17 == (__int64 *)*v15 )
    {
      v17 = *(__int64 **)v14;
      v80 = (__int64 *)*v15;
LABEL_22:
      v19 = *(_QWORD *)(a1 + 8);
      v77 = (__int64)v17;
      v20 = sub_1649960(a2);
      v21 = *(_QWORD *)(a2 - 72);
      v81 = v20;
      v82 = v22;
      v85 = 773;
      v83 = &v81;
      v84 = ".v";
      v23 = (const char *)sub_1707C10(v19, v21, v77, v16, (__int64 *)&v83, a2);
      goto LABEL_23;
    }
    return 0;
  }
  v79 = *(const char **)(v14 + 24);
  v62 = *(_QWORD *)(a1 + 8);
  v76 = *v15;
  v63 = sub_1649960(a2);
  v64 = *(_QWORD *)(a2 - 72);
  v81 = v63;
  v82 = v65;
  v85 = 773;
  v83 = &v81;
  v84 = ".v";
  v66 = sub_1707C10(v62, v64, (__int64)v80, v76, (__int64 *)&v83, a2);
  v23 = v79;
  v80 = v66;
LABEL_23:
  v24 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned int)(v24 - 35) > 0x11 )
  {
    v78 = v23;
    v51 = *(_QWORD *)(a3 + 64);
    v52 = sub_15FA300(a3);
    v53 = v78;
    if ( v52 && (v54 = sub_15FA300(a4), v53 = v78, v54) )
    {
      v81 = v78;
      v85 = 257;
      if ( !v51 )
      {
        v75 = *v80;
        if ( *(_BYTE *)(*v80 + 8) == 16 )
          v75 = **(_QWORD **)(v75 + 16);
        v51 = *(_QWORD *)(v75 + 24);
      }
      v68 = sub_1648A60(72, 2u);
      v13 = (__int64)v68;
      if ( v68 )
      {
        v69 = (__int64)(v68 - 6);
        v70 = *v80;
        if ( *(_BYTE *)(*v80 + 8) == 16 )
          v70 = **(_QWORD **)(v70 + 16);
        v71 = *(_DWORD *)(v70 + 8);
        v72 = (__int64 *)sub_15F9F50(v51, (__int64)&v81, 1);
        v73 = (__int64 *)sub_1646BA0(v72, v71 >> 8);
        v74 = *v80;
        if ( *(_BYTE *)(*v80 + 8) == 16 || (v74 = *(_QWORD *)v81, *(_BYTE *)(*(_QWORD *)v81 + 8LL) == 16) )
          v73 = sub_16463B0(v73, *(_QWORD *)(v74 + 32));
        sub_15F1EA0(v13, (__int64)v73, 32, v69, 2, 0);
        *(_QWORD *)(v13 + 56) = v51;
        *(_QWORD *)(v13 + 64) = sub_15F9F50(v51, (__int64)&v81, 1);
        sub_15F9CE0(v13, (__int64)v80, (__int64 *)&v81, 1, (__int64)&v83);
      }
      sub_15FA2E0(v13, 1);
    }
    else
    {
      v81 = v53;
      v85 = 257;
      if ( !v51 )
      {
        v67 = *v80;
        if ( *(_BYTE *)(*v80 + 8) == 16 )
          v67 = **(_QWORD **)(v67 + 16);
        v51 = *(_QWORD *)(v67 + 24);
      }
      v55 = sub_1648A60(72, 2u);
      v13 = (__int64)v55;
      if ( v55 )
      {
        v56 = (__int64)(v55 - 6);
        v57 = *v80;
        if ( *(_BYTE *)(*v80 + 8) == 16 )
          v57 = **(_QWORD **)(v57 + 16);
        v58 = *(_DWORD *)(v57 + 8);
        v59 = (__int64 *)sub_15F9F50(v51, (__int64)&v81, 1);
        v60 = (__int64 *)sub_1646BA0(v59, v58 >> 8);
        v61 = *v80;
        if ( *(_BYTE *)(*v80 + 8) == 16 || (v61 = *(_QWORD *)v81, *(_BYTE *)(*(_QWORD *)v81 + 8LL) == 16) )
          v60 = sub_16463B0(v60, *(_QWORD *)(v61 + 32));
        sub_15F1EA0(v13, (__int64)v60, 32, v56, 2, 0);
        *(_QWORD *)(v13 + 56) = v51;
        *(_QWORD *)(v13 + 64) = sub_15F9F50(v51, (__int64)&v81, 1);
        sub_15F9CE0(v13, (__int64)v80, (__int64 *)&v81, 1, (__int64)&v83);
      }
    }
  }
  else
  {
    v85 = 257;
    return sub_15FB440(v24 - 24, v80, (__int64)v23, (__int64)&v83, 0);
  }
  return v13;
}
