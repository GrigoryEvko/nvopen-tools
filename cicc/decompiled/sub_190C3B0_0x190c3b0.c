// Function: sub_190C3B0
// Address: 0x190c3b0
//
__int64 __fastcall sub_190C3B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 v10; // rax
  int v11; // eax
  unsigned __int64 v12; // r12
  _QWORD *v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  char v18; // al
  _QWORD *v19; // rdi
  char v20; // al
  bool v21; // bl
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v28; // r9
  char v29; // dl
  bool v30; // r10
  __int64 *v31; // r14
  __int64 v32; // rax
  bool v33; // al
  bool v34; // r8
  __int64 v35; // rax
  int v36; // eax
  bool v37; // al
  bool v38; // r8
  char v39; // al
  bool v40; // bl
  bool v41; // al
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rdx
  __int64 v50; // rax
  bool v51; // [rsp+Eh] [rbp-52h]
  bool v52; // [rsp+Fh] [rbp-51h]
  bool v53; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v54; // [rsp+Fh] [rbp-51h]
  __int64 *v55; // [rsp+10h] [rbp-50h]
  unsigned __int64 v56; // [rsp+10h] [rbp-50h]
  bool v57; // [rsp+10h] [rbp-50h]
  int v58; // [rsp+10h] [rbp-50h]
  unsigned __int8 v59; // [rsp+18h] [rbp-48h]
  unsigned __int8 v60; // [rsp+18h] [rbp-48h]
  bool v61; // [rsp+18h] [rbp-48h]
  bool v62; // [rsp+18h] [rbp-48h]
  unsigned __int8 v63; // [rsp+18h] [rbp-48h]
  unsigned __int8 v64; // [rsp+18h] [rbp-48h]
  unsigned __int64 v65; // [rsp+18h] [rbp-48h]
  bool v66; // [rsp+18h] [rbp-48h]
  unsigned __int64 v67; // [rsp+18h] [rbp-48h]
  unsigned __int64 v69; // [rsp+20h] [rbp-40h]
  __int64 *v70; // [rsp+20h] [rbp-40h]
  unsigned __int64 v71; // [rsp+20h] [rbp-40h]
  __int64 v72; // [rsp+28h] [rbp-38h]
  unsigned __int8 v73; // [rsp+28h] [rbp-38h]

  v10 = sub_15F2050((__int64)a2);
  v72 = sub_1632FA0(v10);
  v11 = a3 & 7;
  if ( v11 == 1 )
  {
    v28 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    v29 = *(_BYTE *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v30 = a4 != 0;
    if ( a4 != 0 && v29 == 55 )
    {
      v51 = a4 != 0 && v29 == 55;
      v56 = a3 & 0xFFFFFFFFFFFFFFF8LL;
      v62 = sub_15F32D0((__int64)a2);
      v37 = sub_15F32D0(a3 & 0xFFFFFFFFFFFFFFF8LL);
      v28 = a3 & 0xFFFFFFFFFFFFFFF8LL;
      v30 = a4 != 0;
      v38 = v51;
      if ( (unsigned __int8)v62 <= (unsigned __int8)v37 )
      {
        if ( a6 )
        {
          if ( *(_BYTE *)(**(_QWORD **)(v56 - 48) + 8LL) == 13 )
          {
            v50 = sub_190A350(a3 & 0xFFFFFFFFFFFFFFF8LL, a2, v72);
            v28 = a3 & 0xFFFFFFFFFFFFFFF8LL;
            v30 = a4 != 0;
            v38 = v51;
            if ( v50 )
              goto LABEL_63;
          }
        }
        v57 = v38;
        v65 = v28;
        v53 = v30;
        v43 = sub_1B6F4A0(*a2, a4, v28, v72);
        v28 = v65;
        v17 = v57;
        if ( v43 != -1 )
        {
          v49 = *(_QWORD *)(v65 - 48);
          *(_DWORD *)(a5 + 8) = v43;
          *(_QWORD *)a5 = v49 & 0xFFFFFFFFFFFFFFF9LL;
          return v17;
        }
        v29 = *(_BYTE *)(v65 + 16);
        v30 = v53;
      }
      else
      {
        v29 = *(_BYTE *)(v56 + 16);
      }
    }
    if ( v29 == 54 )
    {
      if ( !v30 || a2 == (__int64 *)v28 )
      {
LABEL_24:
        v31 = *(__int64 **)(a1 + 104);
        v32 = sub_15E0530(*v31);
        if ( sub_1602790(v32)
          || (v47 = sub_15E0530(*v31),
              v48 = sub_16033E0(v47),
              (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v48 + 32LL))(v48, "gvn", 3))
          || (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v48 + 40LL))(v48, "gvn", 3)
          || (v17 = (*(__int64 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v48 + 24LL))(v48, "gvn", 3),
              (_BYTE)v17) )
        {
          sub_190C060(a2, a3, *(_QWORD *)(a1 + 24), *(_QWORD **)(a1 + 104));
          return 0;
        }
        return v17;
      }
      v52 = v30 && a2 != (__int64 *)v28;
      v55 = (__int64 *)v28;
      v61 = sub_15F32D0((__int64)a2);
      v33 = sub_15F32D0((__int64)v55);
      v28 = (unsigned __int64)v55;
      v34 = v52;
      if ( (unsigned __int8)v61 <= (unsigned __int8)v33 )
      {
        if ( a6 )
        {
          if ( *(_BYTE *)(*v55 + 8) == 13 )
          {
            v50 = sub_190A170(v55, a2, v72);
            v28 = (unsigned __int64)v55;
            v34 = v52;
            if ( v50 )
              goto LABEL_63;
          }
        }
        v66 = v34;
        v70 = (__int64 *)v28;
        v45 = sub_1B6F750(*a2, a4, v28, v72);
        v28 = (unsigned __int64)v70;
        if ( v45 != -1 )
        {
          v54 = v66;
          v67 = (unsigned __int64)v70;
          v58 = v45;
          v71 = v45 + ((unsigned __int64)(sub_127FA20(v72, *a2) + 7) >> 3);
          v46 = sub_127FA20(v72, *(_QWORD *)v67);
          v28 = v67;
          v17 = v54;
          if ( v71 <= (unsigned __int64)(v46 + 7) >> 3 )
          {
            *(_DWORD *)(a5 + 8) = v58;
            *(_QWORD *)a5 = v67 | 2;
            return v17;
          }
        }
      }
      v29 = *(_BYTE *)(v28 + 16);
    }
    if ( v29 == 78 )
    {
      v35 = *(_QWORD *)(v28 - 24);
      v69 = v28;
      if ( !*(_BYTE *)(v35 + 16)
        && (*(_BYTE *)(v35 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v35 + 36) - 133) <= 4
        && ((1LL << (*(_BYTE *)(v35 + 36) + 123)) & 0x15) != 0 )
      {
        if ( a4 )
        {
          if ( !sub_15F32D0((__int64)a2) )
          {
            v36 = sub_1B6FD70(*a2, a4, v69, v72);
            if ( v36 != -1 )
            {
              *(_DWORD *)(a5 + 8) = v36;
              v17 = 1;
              *(_QWORD *)a5 = v69 | 4;
              return v17;
            }
          }
        }
      }
    }
    goto LABEL_24;
  }
  if ( (a3 & 6) != 0 && v11 != 2 )
    BUG();
  v12 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)(v12 + 16) == 53
    || (unsigned __int8)sub_140B0A0(v12, *(_QWORD **)(a1 + 32), 0)
    || *(_BYTE *)(v12 + 16) == 78
    && (v25 = *(_QWORD *)(v12 - 24), !*(_BYTE *)(v25 + 16))
    && (*(_BYTE *)(v25 + 33) & 0x20) != 0
    && *(_DWORD *)(v25 + 36) == 117 )
  {
    v26 = sub_1599EF0((__int64 **)*a2);
    *(_DWORD *)(a5 + 8) = 0;
    v17 = 1;
    *(_QWORD *)a5 = v26 & 0xFFFFFFFFFFFFFFF9LL;
    return v17;
  }
  v13 = *(_QWORD **)(a1 + 32);
  v14 = sub_140B100(v12, v13, 0);
  v17 = v14;
  if ( (_BYTE)v14 )
  {
    v73 = v14;
    v44 = sub_15A06D0((__int64 **)*a2, (__int64)v13, v15, v16);
    *(_DWORD *)(a5 + 8) = 0;
    v17 = v73;
    *(_QWORD *)a5 = v44 & 0xFFFFFFFFFFFFFFF9LL;
    return v17;
  }
  v18 = *(_BYTE *)(v12 + 16);
  if ( v18 != 55 )
  {
    if ( v18 != 54 )
      return v17;
    if ( *a2 != *(_QWORD *)v12 )
    {
      v63 = v17;
      v39 = sub_1B6EFF0(v12, *a2, v72);
      v17 = v63;
      if ( !v39 )
        return v17;
    }
    v64 = v17;
    v40 = sub_15F32D0(v12);
    v41 = sub_15F32D0((__int64)a2);
    v17 = v64;
    if ( (unsigned __int8)v40 < (unsigned __int8)v41 )
      return v17;
    v42 = *(_QWORD *)v12;
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 13 )
    {
      if ( a6 )
      {
        v50 = sub_190A170((__int64 *)v12, a2, v72);
        if ( v50 )
          goto LABEL_63;
        v42 = *(_QWORD *)v12;
        v17 = v64;
      }
      if ( *a2 != v42 )
        return v17;
    }
    *(_DWORD *)(a5 + 8) = 0;
    v17 = 1;
    *(_QWORD *)a5 = v12 | 2;
    return v17;
  }
  v19 = *(_QWORD **)(v12 - 48);
  if ( *a2 != *v19 )
  {
    v59 = v17;
    v20 = sub_1B6EFF0(v19, *a2, v72);
    v17 = v59;
    if ( !v20 )
      return v17;
  }
  v60 = v17;
  v21 = sub_15F32D0(v12);
  v22 = sub_15F32D0((__int64)a2);
  v17 = v60;
  if ( (unsigned __int8)v21 < (unsigned __int8)v22 )
    return v17;
  v23 = *(_QWORD *)(v12 - 48);
  v24 = *(_QWORD *)v23;
  if ( *(_BYTE *)(*(_QWORD *)v23 + 8LL) != 13 )
  {
LABEL_15:
    *(_DWORD *)(a5 + 8) = 0;
    v17 = 1;
    *(_QWORD *)a5 = v23 & 0xFFFFFFFFFFFFFFF9LL;
    return v17;
  }
  if ( !a6 )
    goto LABEL_14;
  v50 = sub_190A350(v12, a2, v72);
  if ( v50 )
  {
LABEL_63:
    *(_DWORD *)(a5 + 8) = 0;
    v17 = a6;
    *(_QWORD *)a5 = v50 & 0xFFFFFFFFFFFFFFF9LL;
    return v17;
  }
  v23 = *(_QWORD *)(v12 - 48);
  v17 = v60;
  v24 = *(_QWORD *)v23;
LABEL_14:
  if ( *a2 == v24 )
    goto LABEL_15;
  return v17;
}
