// Function: sub_13DE4E0
// Address: 0x13de4e0
//
unsigned __int8 *__fastcall sub_13DE4E0(__int64 a1, __int64 a2, char a3, char a4, _QWORD *a5, int a6)
{
  __int64 v6; // r14
  unsigned __int8 v10; // al
  __int64 v11; // r9
  unsigned __int8 v13; // al
  char v14; // al
  unsigned int v15; // r15d
  int v16; // eax
  bool v17; // al
  unsigned __int8 v18; // al
  unsigned int v19; // eax
  __int64 v20; // rsi
  unsigned int v21; // r13d
  int v22; // eax
  unsigned __int8 *v23; // r9
  char v24; // al
  char v25; // al
  char v26; // al
  __int64 v27; // r15
  __int64 v28; // rdi
  char v29; // al
  __int64 v30; // rax
  unsigned int v31; // r15d
  int v32; // eax
  unsigned __int8 *v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rcx
  _QWORD *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned int v41; // r15d
  __int64 v42; // rax
  char v43; // cl
  int v44; // eax
  __int64 v45; // rax
  unsigned int v46; // edx
  __int64 v47; // rdi
  unsigned int v48; // r13d
  int v49; // eax
  char v50; // al
  unsigned int v51; // r15d
  __int64 v52; // rax
  char v53; // cl
  unsigned int v54; // esi
  __int64 v55; // r8
  unsigned int v56; // ecx
  int v57; // eax
  __int64 v58; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v59; // [rsp+10h] [rbp-80h]
  int v60; // [rsp+18h] [rbp-78h]
  _QWORD *v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  __int64 v63; // [rsp+20h] [rbp-70h]
  int v64; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v65; // [rsp+20h] [rbp-70h]
  int v66; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v68; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v69; // [rsp+28h] [rbp-68h]
  __int64 v70; // [rsp+38h] [rbp-58h] BYREF
  __int64 v71; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v72[8]; // [rsp+50h] [rbp-40h] BYREF

  v6 = a1;
  v10 = *(_BYTE *)(a1 + 16);
  if ( v10 > 0x10u )
    goto LABEL_5;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
    v11 = sub_14D6F90(11, a1, a2, *a5);
    if ( v11 )
      return (unsigned __int8 *)v11;
LABEL_5:
    v13 = *(_BYTE *)(a2 + 16);
    if ( v13 == 9 )
      return (unsigned __int8 *)a2;
    v11 = a1;
    v6 = a2;
    if ( v13 > 0x10u )
      goto LABEL_7;
    goto LABEL_10;
  }
  v11 = a2;
  if ( v10 == 9 )
    return (unsigned __int8 *)v6;
LABEL_10:
  v62 = v11;
  v14 = sub_1593BB0(v6);
  v11 = v62;
  if ( v14 )
    return (unsigned __int8 *)v11;
  if ( *(_BYTE *)(v6 + 16) == 13 )
  {
    v15 = *(_DWORD *)(v6 + 32);
    if ( v15 <= 0x40 )
    {
      v17 = *(_QWORD *)(v6 + 24) == 0;
    }
    else
    {
      v16 = sub_16A57B0(v6 + 24);
      v11 = v62;
      v17 = v15 == v16;
    }
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
      goto LABEL_7;
    v30 = sub_15A1020(v6);
    v11 = v62;
    if ( !v30 || *(_BYTE *)(v30 + 16) != 13 )
    {
      v60 = *(_QWORD *)(*(_QWORD *)v6 + 32LL);
      if ( !v60 )
        return (unsigned __int8 *)v11;
      v41 = 0;
      while ( 1 )
      {
        v63 = v11;
        v42 = sub_15A0A60(v6, v41);
        v11 = v63;
        if ( !v42 )
          goto LABEL_7;
        v43 = *(_BYTE *)(v42 + 16);
        if ( v43 != 9 )
        {
          if ( v43 != 13 )
            goto LABEL_7;
          if ( *(_DWORD *)(v42 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v42 + 24) )
              goto LABEL_7;
          }
          else
          {
            v58 = v63;
            v64 = *(_DWORD *)(v42 + 32);
            v44 = sub_16A57B0(v42 + 24);
            v11 = v58;
            if ( v64 != v44 )
              goto LABEL_7;
          }
        }
        if ( v60 == ++v41 )
          return (unsigned __int8 *)v11;
      }
    }
    v31 = *(_DWORD *)(v30 + 32);
    if ( v31 <= 0x40 )
    {
      v17 = *(_QWORD *)(v30 + 24) == 0;
    }
    else
    {
      v32 = sub_16A57B0(v30 + 24);
      v11 = v62;
      v17 = v31 == v32;
    }
  }
  if ( v17 )
    return (unsigned __int8 *)v11;
LABEL_7:
  v61 = (_QWORD *)v11;
  if ( !(unsigned __int8)sub_14B0710(v11, v6, 0) )
  {
    v25 = *(_BYTE *)(v6 + 16);
    v70 = 0;
    if ( v25 == 37 )
    {
      v33 = *(unsigned __int8 **)(v6 - 48);
      if ( v33 )
      {
        v34 = *(_QWORD **)(v6 - 24);
        v70 = *(_QWORD *)(v6 - 48);
        if ( v34 )
        {
          if ( v34 == v61 )
            return v33;
        }
      }
    }
    else if ( v25 == 5 && *(_WORD *)(v6 + 18) == 13 )
    {
      v36 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
      v37 = *(_QWORD *)(v6 - 24 * v36);
      if ( v37 )
      {
        v70 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v38 = *(_QWORD **)(v6 + 24 * (1 - v36));
        if ( v38 )
        {
          if ( v38 == v61 )
            return (unsigned __int8 *)v37;
        }
      }
    }
    v26 = *((_BYTE *)v61 + 16);
    if ( v26 == 37 )
    {
      v33 = (unsigned __int8 *)*(v61 - 6);
      if ( v33 )
      {
        v35 = *(v61 - 3);
        v70 = *(v61 - 6);
        if ( v35 )
        {
          if ( v35 == v6 )
            return v33;
        }
      }
    }
    else if ( v26 == 5 && *((_WORD *)v61 + 9) == 13 )
    {
      v39 = *((_DWORD *)v61 + 5) & 0xFFFFFFF;
      v37 = v61[-3 * v39];
      if ( v37 )
      {
        v70 = v61[-3 * (*((_DWORD *)v61 + 5) & 0xFFFFFFF)];
        v40 = v61[3 * (1 - v39)];
        if ( v6 == v40 )
        {
          if ( v40 )
            return (unsigned __int8 *)v37;
        }
      }
    }
    v71 = v6;
    v27 = *v61;
    if ( sub_13D1F50(&v71, (__int64)v61) )
      return (unsigned __int8 *)sub_15A04A0(v27);
    v72[0] = v61;
    if ( sub_13D1F50((__int64 *)v72, v6) )
      return (unsigned __int8 *)sub_15A04A0(v27);
    v23 = (unsigned __int8 *)v61;
    if ( !a3 && !a4 )
    {
LABEL_34:
      if ( !a6 )
        return sub_13DDF20(11, v23, (unsigned __int8 *)v6, a5, a6);
      v28 = *(_QWORD *)v23;
      if ( *(_BYTE *)(*(_QWORD *)v23 + 8LL) == 16 )
        v28 = **(_QWORD **)(v28 + 16);
      v69 = v23;
      v29 = sub_1642F90(v28, 1);
      v23 = v69;
      if ( !v29 )
        return sub_13DDF20(11, v23, (unsigned __int8 *)v6, a5, a6);
      v33 = sub_13DE280(v69, (unsigned __int8 *)v6, a5, a6 - 1);
      v23 = v69;
      if ( !v33 )
        return sub_13DDF20(11, v23, (unsigned __int8 *)v6, a5, a6);
      return v33;
    }
    v18 = *(_BYTE *)(v6 + 16);
    if ( v18 == 13 )
    {
      v19 = *(_DWORD *)(v6 + 32);
      v20 = *(_QWORD *)(v6 + 24);
      v21 = v19 - 1;
      if ( v19 <= 0x40 )
      {
        if ( v20 != 1LL << v21 )
          goto LABEL_21;
      }
      else
      {
        if ( (*(_QWORD *)(v20 + 8LL * (v21 >> 6)) & (1LL << v21)) == 0 )
          goto LABEL_21;
        v22 = sub_16A58A0(v6 + 24);
        v23 = (unsigned __int8 *)v61;
        if ( v21 != v22 )
          goto LABEL_21;
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 || v18 > 0x10u )
        goto LABEL_21;
      v45 = sub_15A1020(v6);
      v23 = (unsigned __int8 *)v61;
      if ( v45 && *(_BYTE *)(v45 + 16) == 13 )
      {
        v46 = *(_DWORD *)(v45 + 32);
        v47 = *(_QWORD *)(v45 + 24);
        v48 = v46 - 1;
        if ( v46 <= 0x40 )
        {
          if ( v47 != 1LL << v48 )
            goto LABEL_21;
        }
        else
        {
          if ( (*(_QWORD *)(v47 + 8LL * (v48 >> 6)) & (1LL << v48)) == 0 )
            goto LABEL_21;
          v49 = sub_16A58A0(v45 + 24);
          v23 = (unsigned __int8 *)v61;
          if ( v49 != v48 )
            goto LABEL_21;
        }
      }
      else
      {
        v51 = 0;
        v66 = *(_DWORD *)(*(_QWORD *)v6 + 32LL);
        while ( v66 != v51 )
        {
          v59 = v23;
          v52 = sub_15A0A60(v6, v51);
          v23 = v59;
          if ( !v52 )
            goto LABEL_21;
          v53 = *(_BYTE *)(v52 + 16);
          if ( v53 != 9 )
          {
            if ( v53 != 13 )
              goto LABEL_21;
            v54 = *(_DWORD *)(v52 + 32);
            v55 = *(_QWORD *)(v52 + 24);
            v56 = v54 - 1;
            if ( v54 <= 0x40 )
            {
              if ( v55 != 1LL << v56 )
                goto LABEL_21;
            }
            else
            {
              if ( (*(_QWORD *)(v55 + 8LL * (v56 >> 6)) & (1LL << v56)) == 0 )
                goto LABEL_21;
              v57 = sub_16A58A0(v52 + 24);
              v23 = v59;
              if ( v54 - 1 != v57 )
                goto LABEL_21;
            }
          }
          ++v51;
        }
      }
    }
    v65 = v23;
    v72[0] = &v70;
    v50 = sub_13D21C0(v72, (__int64)v23);
    v23 = v65;
    if ( v50 )
      return (unsigned __int8 *)v70;
LABEL_21:
    if ( !a4 )
      goto LABEL_34;
    v68 = v23;
    v24 = sub_13CC520(v6);
    v23 = v68;
    if ( !v24 )
      goto LABEL_34;
    return (unsigned __int8 *)v6;
  }
  return (unsigned __int8 *)sub_15A06D0(*v61);
}
