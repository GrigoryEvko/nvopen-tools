// Function: sub_1668250
// Address: 0x1668250
//
void __fastcall sub_1668250(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 **v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 *v6; // r13
  int v7; // eax
  unsigned int v8; // r9d
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _QWORD *v12; // r15
  __int64 v13; // r14
  _QWORD *v14; // r14
  int v15; // r15d
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // r13
  _QWORD *v24; // rdi
  const char *v25; // rax
  unsigned int v26; // eax
  __int64 v27; // r13
  __int64 v28; // rax
  const char *v29; // rax
  __int64 *v30; // rsi
  const char *v31; // rax
  int v32; // r15d
  _QWORD *v33; // rdi
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // r15
  unsigned __int64 v38; // r13
  __int64 *v39; // rbx
  char v40; // r12
  unsigned int v41; // r14d
  char v42; // r15
  char v43; // al
  char v44; // al
  unsigned int v45; // eax
  char v46; // di
  unsigned int v47; // ebx
  char v48; // r12
  __int64 v49; // r14
  char v50; // al
  __int64 *v51; // rax
  const char *v52; // rax
  unsigned __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rcx
  char v57; // dl
  __int64 v58; // rdx
  __int64 *v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rcx
  int v62; // esi
  unsigned __int64 v63; // rdi
  unsigned int v64; // eax
  char v65; // r12
  __int64 v66; // r13
  __int64 v67; // rbx
  _QWORD *v68; // r15
  __int64 v69; // rax
  unsigned __int64 v70; // r14
  unsigned int *v71; // rax
  int v72; // edi
  __int64 v73; // r9
  unsigned __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // rax
  __int64 *v78; // rsi
  __int64 v79; // r13
  _BYTE *v80; // rax
  bool v81; // zf
  _QWORD *v82; // [rsp+0h] [rbp-A0h]
  __int64 v83; // [rsp+8h] [rbp-98h]
  __int64 *v84; // [rsp+10h] [rbp-90h]
  unsigned __int64 v85; // [rsp+18h] [rbp-88h]
  __int64 *v86; // [rsp+20h] [rbp-80h]
  __int64 v87[2]; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v88; // [rsp+38h] [rbp-68h] BYREF
  __int64 v89; // [rsp+40h] [rbp-60h] BYREF
  __int64 v90; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v91[2]; // [rsp+50h] [rbp-50h] BYREF
  char v92; // [rsp+60h] [rbp-40h]
  char v93; // [rsp+61h] [rbp-3Fh]

  v2 = a1;
  v87[0] = a2;
  v3 = (__int64 **)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v88 = (__int64 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a2 & 4) == 0 )
    v3 = (__int64 **)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  v4 = **v3;
  if ( *(_BYTE *)(v4 + 8) != 15 )
  {
    v93 = 1;
    v25 = "Called function must be a pointer!";
    goto LABEL_29;
  }
  v5 = *(_QWORD *)(v4 + 24);
  if ( *(_BYTE *)(v5 + 8) != 12 )
  {
    v93 = 1;
    v25 = "Called function is not pointer to function type!";
    goto LABEL_29;
  }
  if ( v5 != *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 64) )
  {
    v93 = 1;
    v25 = "Called function is not the same type as the call!";
    goto LABEL_29;
  }
  v6 = v87;
  if ( *(_DWORD *)(v5 + 8) >> 8 )
  {
    v26 = sub_165AFC0(v87);
    v8 = *(_DWORD *)(v5 + 12) - 1;
    if ( v26 >= v8 )
      goto LABEL_8;
    v93 = 1;
    v25 = "Called function requires more parameters than were provided!";
LABEL_29:
    v91[0] = v25;
    v92 = 3;
    sub_1654980(a1, (__int64)v91, (__int64 *)&v88);
    return;
  }
  v7 = sub_165AFC0(v87);
  v8 = *(_DWORD *)(v5 + 12) - 1;
  if ( v7 != v8 )
  {
    v93 = 1;
    v25 = "Incorrect number of arguments passed to called function!";
    goto LABEL_29;
  }
LABEL_8:
  v9 = 0;
  LODWORD(v10) = 0;
  v11 = v87[0] & 0xFFFFFFFFFFFFFFF8LL;
  while ( (_DWORD)v10 != v8 )
  {
    v12 = *(_QWORD **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) + v9);
    v10 = (unsigned int)(v10 + 1);
    v9 += 24;
    v13 = *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8 * v10);
    if ( v13 != *v12 )
    {
      v93 = 1;
      v91[0] = "Call parameter type does not match function signature!";
      v92 = 3;
      sub_164FF40(a1, (__int64)v91);
      if ( !*a1 )
        return;
      sub_164FA80(a1, (__int64)v12);
      if ( v13 )
        sub_164ECF0(*a1, v13);
      goto LABEL_38;
    }
  }
  v14 = v91;
  v89 = *(_QWORD *)(v11 + 56);
  v15 = sub_165AFC0(v87);
  v91[0] = v89;
  if ( (unsigned int)sub_15601D0((__int64)v91) > v15 + 2 )
  {
    v93 = 1;
    v31 = "Attribute after last parameter!";
    goto LABEL_51;
  }
  v86 = &v89;
  if ( (unsigned __int8)sub_1560260(&v89, -1, 47) )
  {
    v16 = (__int64 *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 72);
    if ( (v87[0] & 4) != 0 )
      v16 = (__int64 *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
    v17 = sub_1649C60(*v16);
    if ( *(_BYTE *)(v17 + 16) || !(unsigned __int8)sub_1560180(v17 + 112, 47) )
    {
      v93 = 1;
      v31 = "speculatable attribute may not apply to call sites";
      goto LABEL_51;
    }
  }
  sub_16595D0(a1, v5, v89, v88);
  v18 = sub_1389B50(v87);
  if ( v18 == (v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF) )
    goto LABEL_18;
  v32 = sub_165AFC0(v87) - 1;
  v85 = v87[0] & 0xFFFFFFFFFFFFFFF8LL;
  v33 = (_QWORD *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v87[0] & 4) == 0 )
  {
    if ( (unsigned __int8)sub_1560290(v33, v32, 11) )
      goto LABEL_58;
    v34 = *(_QWORD *)(v85 - 72);
    if ( *(_BYTE *)(v34 + 16) )
      goto LABEL_18;
    goto LABEL_57;
  }
  if ( (unsigned __int8)sub_1560290(v33, v32, 11) )
  {
LABEL_58:
    v35 = sub_165B7C0(v87);
    v36 = sub_164A820(*(_QWORD *)(v35 + 24LL * (unsigned int)(*(_DWORD *)(v5 + 12) - 2)));
    v37 = (__int64 *)v36;
    if ( *(_BYTE *)(v36 + 16) == 53 && (*(_BYTE *)(v36 + 18) & 0x20) == 0 )
    {
      v93 = 1;
      v91[0] = "inalloca argument for call has mismatched alloca";
      v92 = 3;
      sub_164FF40(v2, (__int64)v91);
      v30 = v37;
      if ( !*v2 )
        return;
LABEL_45:
      sub_164FA80(v2, (__int64)v30);
      goto LABEL_38;
    }
    goto LABEL_18;
  }
  v34 = *(_QWORD *)(v85 - 24);
  if ( !*(_BYTE *)(v34 + 16) )
  {
LABEL_57:
    v91[0] = *(_QWORD *)(v34 + 112);
    if ( (unsigned __int8)sub_1560290(v91, v32, 11) )
      goto LABEL_58;
  }
LABEL_18:
  v19 = *(_DWORD *)(v5 + 12);
  if ( v19 == 1 )
  {
    if ( *(_DWORD *)(v5 + 8) >> 8 )
      goto LABEL_128;
    goto LABEL_88;
  }
  v85 = v5;
  v84 = v87;
  v20 = 0;
  v21 = (unsigned int)(v19 - 1);
  do
  {
    v23 = v87[0] & 0xFFFFFFFFFFFFFFF8LL;
    v24 = (_QWORD *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
    if ( (v87[0] & 4) != 0 )
    {
      if ( !(unsigned __int8)sub_1560290(v24, v20, 54) )
      {
        v22 = *(_QWORD *)(v23 - 24);
        if ( *(_BYTE *)(v22 + 16) )
          goto LABEL_23;
LABEL_22:
        v91[0] = *(_QWORD *)(v22 + 112);
        if ( !(unsigned __int8)sub_1560290(v91, v20, 54) )
          goto LABEL_23;
      }
    }
    else if ( !(unsigned __int8)sub_1560290(v24, v20, 54) )
    {
      v22 = *(_QWORD *)(v23 - 72);
      if ( *(_BYTE *)(v22 + 16) )
        goto LABEL_23;
      goto LABEL_22;
    }
    v27 = *(_QWORD *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL)
                    + 24 * (v20 - (*(_DWORD *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
    v28 = sub_164A820(v27);
    if ( *(_BYTE *)(v28 + 16) == 53 )
    {
      if ( (*(_BYTE *)(v28 + 18) & 0x40) == 0 )
      {
        v86 = (__int64 *)v28;
        v93 = 1;
        v91[0] = "swifterror argument for call has mismatched alloca";
        v92 = 3;
        sub_164FF40(v2, (__int64)v91);
        if ( !*v2 )
          return;
        v30 = v86;
        goto LABEL_45;
      }
    }
    else
    {
      if ( *(_BYTE *)(v27 + 16) != 17 )
      {
        v86 = (__int64 *)v27;
        v29 = "swifterror argument should come from an alloca or parameter";
        v93 = 1;
        goto LABEL_43;
      }
      if ( !(unsigned __int8)sub_15E02D0(v27) )
      {
        v86 = (__int64 *)v27;
        v29 = "swifterror argument for call has mismatched parameter";
        v93 = 1;
LABEL_43:
        v91[0] = v29;
        v92 = 3;
        sub_164FF40(v2, (__int64)v91);
        if ( !*v2 )
          return;
        v30 = v86;
        goto LABEL_45;
      }
    }
LABEL_23:
    ++v20;
  }
  while ( v21 != v20 );
  v5 = v85;
  v6 = v84;
  if ( *(_DWORD *)(v85 + 8) >> 8 )
  {
    if ( *(_DWORD *)(v85 + 12) != 1 )
    {
      v83 = (__int64)v84;
      v38 = v85;
      v39 = v86;
      v84 = v2;
      v40 = 0;
      v82 = v91;
      v41 = 0;
      v42 = 0;
      do
      {
        v43 = sub_1560290(v39, v41, 19);
        if ( v43 )
          v40 = v43;
        v44 = sub_1560290(v39, v41, 38);
        if ( v44 )
          v42 = v44;
        ++v41;
        v45 = *(_DWORD *)(v38 + 12) - 1;
      }
      while ( v41 < v45 );
      LOBYTE(v85) = v40;
      v5 = v38;
      v2 = v84;
      v46 = v42;
      v6 = (__int64 *)v83;
      v14 = v82;
      goto LABEL_71;
    }
LABEL_128:
    LOBYTE(v85) = 0;
    v45 = 0;
    v46 = 0;
LABEL_71:
    v83 = v5;
    v47 = v45;
    v84 = v2;
    v48 = v46;
    v82 = v14;
    while ( (unsigned int)sub_165AFC0(v6) > v47 )
    {
      v49 = **(_QWORD **)((v87[0] & 0xFFFFFFFFFFFFFFF8LL)
                        + 24 * (v47 - (unsigned __int64)(*(_DWORD *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
      v90 = sub_1560230(v86, v47);
      sub_1659040((__int64)v84, v90, v49, v88);
      v50 = sub_155EE10((__int64)&v90, 19);
      if ( v50 )
      {
        if ( (_BYTE)v85 )
        {
          v93 = 1;
          v2 = v84;
          v31 = "More than one parameter has attribute nest!";
          v14 = v82;
          goto LABEL_51;
        }
        LOBYTE(v85) = v50;
      }
      if ( (unsigned __int8)sub_155EE10((__int64)&v90, 38) )
      {
        if ( v48 )
        {
          v93 = 1;
          v2 = v84;
          v31 = "More than one parameter has attribute returned!";
          v14 = v82;
          goto LABEL_51;
        }
        v48 = sub_16430A0(v49, **(_QWORD **)(v83 + 16));
        if ( !v48 )
        {
          v93 = 1;
          v2 = v84;
          v31 = "Incompatible argument and return types for 'returned' attribute";
          v14 = v82;
          goto LABEL_51;
        }
      }
      if ( (unsigned __int8)sub_155EE10((__int64)&v90, 53) )
      {
        v93 = 1;
        v2 = v84;
        v31 = "Attribute 'sret' cannot be used for vararg call arguments!";
        v14 = v82;
        goto LABEL_51;
      }
      if ( (unsigned __int8)sub_155EE10((__int64)&v90, 11) && (unsigned int)sub_165AFC0(v6) - 1 != v47 )
      {
        v93 = 1;
        v2 = v84;
        v31 = "inalloca isn't on the last argument!";
        v14 = v82;
        goto LABEL_51;
      }
      ++v47;
    }
    v5 = v83;
    v2 = v84;
    v14 = v82;
  }
LABEL_88:
  v51 = (__int64 *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 72);
  if ( (v87[0] & 4) != 0 )
    v51 = (__int64 *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( *(_BYTE *)(*v51 + 16) || (v52 = sub_1649960(*v51), v53 <= 4) || *(_DWORD *)v52 != 1836477548 || v52[4] != 46 )
  {
    v54 = *(_QWORD *)(v5 + 16);
    v55 = v54 + 8;
    v56 = v54 + 8LL * *(unsigned int *)(v5 + 12);
    if ( v56 != v54 + 8 )
    {
      do
      {
        v57 = *(_BYTE *)(*(_QWORD *)v55 + 8LL);
        if ( v57 == 8 )
        {
          v93 = 1;
          v31 = "Function has metadata parameter but isn't an intrinsic";
          goto LABEL_51;
        }
        if ( v57 == 10 )
        {
          v93 = 1;
          v31 = "Function has token parameter but isn't an intrinsic";
          goto LABEL_51;
        }
        v55 += 8;
      }
      while ( v56 != v55 );
    }
  }
  v58 = v87[0];
  v59 = (__int64 *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 72);
  v60 = (v87[0] >> 2) & 1;
  if ( ((v87[0] >> 2) & 1) != 0 )
    v59 = (__int64 *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v61 = *v59;
  if ( *(_BYTE *)(v61 + 16) )
  {
    if ( *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL) == 10 )
    {
      v93 = 1;
      v91[0] = "Return type cannot be token for indirect call!";
      v92 = 3;
      sub_164FF40(v2, (__int64)v14);
      return;
    }
  }
  else
  {
    v62 = *(_DWORD *)(v61 + 36);
    if ( v62 )
    {
      sub_165FE30((__int64)v2, v62, v87[0]);
      v58 = v87[0];
      v60 = (v87[0] >> 2) & 1;
    }
  }
  v63 = v58 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_BYTE)v60 )
    v64 = sub_165C280(v63);
  else
    v64 = sub_165C2D0(v63);
  if ( !v64 )
  {
LABEL_135:
    v75 = sub_15F2060((__int64)v88);
    if ( !sub_1626D20(v75) || (v76 = sub_165B1A0(v6)) == 0 )
    {
      v78 = v88;
LABEL_139:
      sub_1663F80((__int64)v2, (__int64)v78);
      return;
    }
    v77 = sub_1626D20(v76);
    v78 = v88;
    if ( !v77 || v88[6] )
      goto LABEL_139;
    v79 = *v2;
    v93 = 1;
    v91[0] = "inlinable function call in a function with debug info must have a !dbg location";
    v92 = 3;
    if ( v79 )
    {
      sub_16E2CE0(v14, v79);
      v80 = *(_BYTE **)(v79 + 24);
      if ( (unsigned __int64)v80 >= *(_QWORD *)(v79 + 16) )
      {
        sub_16E7DE0(v79, 10);
      }
      else
      {
        *(_QWORD *)(v79 + 24) = v80 + 1;
        *v80 = 10;
      }
    }
    *((_BYTE *)v2 + 72) |= *((_BYTE *)v2 + 74);
    v81 = *v2 == 0;
    *((_BYTE *)v2 + 73) = 1;
    if ( v81 )
      return;
LABEL_38:
    sub_164FA80(v2, (__int64)v88);
    return;
  }
  v86 = v2;
  v83 = (__int64)v6;
  v65 = 0;
  v66 = 0;
  LOBYTE(v85) = 0;
  v67 = 16LL * v64;
  v68 = v14;
  LOBYTE(v84) = 0;
  while ( 1 )
  {
    v69 = 0;
    v70 = v87[0] & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(char *)((v87[0] & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0 )
      v69 = sub_1648A40(v87[0] & 0xFFFFFFFFFFFFFFF8LL);
    v71 = (unsigned int *)(v66 + v69);
    v72 = *(_DWORD *)(*(_QWORD *)v71 + 8LL);
    v73 = 24LL * v71[2];
    v74 = 0xAAAAAAAAAAAAAAABLL * ((24LL * v71[3] - v73) >> 3);
    if ( !v72 )
    {
      if ( v65 )
      {
        v93 = 1;
        v2 = v86;
        v14 = v68;
        v31 = "Multiple deopt operand bundles";
        goto LABEL_51;
      }
      v65 = 1;
      goto LABEL_111;
    }
    if ( v72 == 2 )
    {
      if ( (_BYTE)v85 )
      {
        v93 = 1;
        v2 = v86;
        v14 = v68;
        v31 = "Multiple gc-transition operand bundles";
        goto LABEL_51;
      }
      LOBYTE(v85) = 1;
      goto LABEL_111;
    }
    if ( v72 != 1 )
      goto LABEL_111;
    if ( (_BYTE)v84 )
      break;
    if ( v74 != 1 )
    {
      v93 = 1;
      v2 = v86;
      v14 = v68;
      v31 = "Expected exactly one funclet bundle operand";
      goto LABEL_51;
    }
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v70 + v73 - 24LL * (*(_DWORD *)(v70 + 20) & 0xFFFFFFF)) + 16LL) - 73) > 1u )
    {
      v93 = 1;
      v2 = v86;
      v14 = v68;
      v31 = "Funclet bundle operands should correspond to a FuncletPadInst";
      goto LABEL_51;
    }
    LOBYTE(v84) = 1;
LABEL_111:
    v66 += 16;
    if ( v67 == v66 )
    {
      v2 = v86;
      v6 = (__int64 *)v83;
      v14 = v68;
      goto LABEL_135;
    }
  }
  v93 = 1;
  v2 = v86;
  v14 = v68;
  v31 = "Multiple funclet operand bundles";
LABEL_51:
  v91[0] = v31;
  v92 = 3;
  sub_1654980(v2, (__int64)v14, (__int64 *)&v88);
}
