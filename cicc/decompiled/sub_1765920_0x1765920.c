// Function: sub_1765920
// Address: 0x1765920
//
_QWORD *__fastcall sub_1765920(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v7; // r15
  __int64 *v8; // r14
  unsigned __int8 v9; // al
  int v10; // r10d
  unsigned int v11; // eax
  __int64 v12; // rcx
  unsigned int v13; // r8d
  unsigned int v14; // r10d
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // r12d
  char v18; // al
  unsigned int v19; // eax
  __int64 v20; // rsi
  unsigned int v21; // ecx
  int v22; // eax
  bool v23; // al
  unsigned int v24; // edi
  unsigned int v25; // edi
  __int16 v26; // bx
  unsigned int v27; // edx
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r13
  _QWORD *v31; // rax
  _QWORD *v32; // r12
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 **v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r13
  _QWORD *v40; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r13
  _QWORD *v45; // rax
  __int64 v46; // rsi
  unsigned int v47; // ebx
  bool v48; // dl
  const void *v49; // r12
  bool v50; // al
  _QWORD *v51; // rax
  unsigned int v52; // eax
  unsigned int v53; // r13d
  bool v54; // al
  unsigned __int64 v55; // rdi
  bool v56; // r12
  _QWORD *v57; // rax
  bool v58; // al
  unsigned int v59; // edi
  unsigned int v60; // [rsp+4h] [rbp-ACh]
  unsigned int v61; // [rsp+8h] [rbp-A8h]
  char v62; // [rsp+8h] [rbp-A8h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  unsigned int v64; // [rsp+18h] [rbp-98h]
  char v65; // [rsp+18h] [rbp-98h]
  unsigned int v66; // [rsp+18h] [rbp-98h]
  char v67; // [rsp+18h] [rbp-98h]
  bool v68; // [rsp+18h] [rbp-98h]
  bool v69; // [rsp+18h] [rbp-98h]
  bool v70; // [rsp+18h] [rbp-98h]
  unsigned __int64 v71; // [rsp+18h] [rbp-98h]
  bool v72; // [rsp+18h] [rbp-98h]
  char v73; // [rsp+2Fh] [rbp-81h] BYREF
  unsigned __int64 v74; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v75; // [rsp+38h] [rbp-78h]
  unsigned __int64 v76; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v77; // [rsp+48h] [rbp-68h]
  unsigned __int64 v78; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v79; // [rsp+58h] [rbp-58h]
  unsigned __int64 v80; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v81; // [rsp+68h] [rbp-48h]
  __int16 v82; // [rsp+70h] [rbp-40h]

  v7 = *(_BYTE **)(a3 - 24);
  v8 = *(__int64 **)(a3 - 48);
  v9 = v7[16];
  if ( v9 == 13 )
  {
    v63 = (__int64)(v7 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
      return 0;
    if ( v9 > 0x10u )
      return 0;
    v42 = sub_15A1020(v7, a2, *(_QWORD *)v7, a4);
    if ( !v42 || *(_BYTE *)(v42 + 16) != 13 )
      return 0;
    v63 = v42 + 24;
  }
  v10 = *(unsigned __int16 *)(a2 + 18);
  v73 = 0;
  v64 = v10 & 0xFFFF7FFF;
  LOBYTE(v11) = sub_1757FA0(v10 & 0xFFFF7FFF, a4, &v73);
  v14 = v64;
  v15 = v11;
  if ( (_BYTE)v11 )
  {
    v33 = *(unsigned int *)(v63 + 8);
    v34 = (unsigned int)(v33 - 1);
    v35 = 1LL << ((unsigned __int8)v33 - 1);
    v36 = *(_QWORD *)v63;
    if ( (unsigned int)v33 > 0x40 )
    {
      v34 = (unsigned int)v34 >> 6;
      v36 = *(_QWORD *)(v36 + 8LL * (unsigned int)v34);
    }
    if ( (v36 & v35) != 0 )
    {
      v37 = (__int64 **)*v8;
      if ( v73 )
      {
        v38 = sub_15A04A0(v37);
        v82 = 257;
        v39 = v38;
        v40 = sub_1648A60(56, 2u);
        v32 = v40;
        if ( v40 )
          sub_17582E0((__int64)v40, 38, (__int64)v8, v39, (__int64)&v80);
      }
      else
      {
        v43 = sub_15A06D0(v37, v33, v35, v34);
        v82 = 257;
        v44 = v43;
        v45 = sub_1648A60(56, 2u);
        v32 = v45;
        if ( v45 )
          sub_17582E0((__int64)v45, 40, (__int64)v8, v44, (__int64)&v80);
      }
    }
    else
    {
      sub_1593B40((_QWORD *)(a2 - 48), (__int64)v8);
      v46 = a3;
      v32 = (_QWORD *)a2;
      sub_170B990(*a1, v46);
    }
    return v32;
  }
  v16 = *(_QWORD *)(a3 + 8);
  if ( v16 )
  {
    if ( !*(_QWORD *)(v16 + 8) )
    {
      v17 = *(_WORD *)(a2 + 18) & 0x7FFF;
      if ( (unsigned int)(v17 - 32) > 1 )
      {
        v61 = v64;
        v65 = v15;
        v18 = sub_13CFF40((__int64 *)v63, a4, v15, v12, v13);
        LOBYTE(v15) = v65;
        v14 = v61;
        if ( v18 )
        {
          v58 = sub_15FF7F0(v17);
          v59 = *(_WORD *)(a2 + 18) & 0x7FFF;
          if ( v58 )
            v26 = sub_15FF470(v59);
          else
            v26 = sub_15FF420(v59);
LABEL_14:
          sub_13A38D0((__int64)&v76, a4);
          v27 = v77;
          if ( v77 > 0x40 )
          {
            sub_16A8F00((__int64 *)&v76, (__int64 *)v63);
            v27 = v77;
            v28 = v76;
          }
          else
          {
            v28 = *(_QWORD *)v63 ^ v76;
            v76 = v28;
          }
          v79 = v27;
          v78 = v28;
          v77 = 0;
          v29 = sub_15A1070(*v8, (__int64)&v78);
          v82 = 257;
          v30 = v29;
          v31 = sub_1648A60(56, 2u);
          v32 = v31;
          if ( v31 )
            sub_17582E0((__int64)v31, v26, (__int64)v8, v30, (__int64)&v80);
          sub_135E100((__int64 *)&v78);
          sub_135E100((__int64 *)&v76);
          return v32;
        }
        v19 = *(_DWORD *)(v63 + 8);
        v20 = *(_QWORD *)v63;
        v21 = v19 - 1;
        if ( v19 <= 0x40 )
        {
          if ( v20 == (1LL << v21) - 1 )
            goto LABEL_11;
        }
        else
        {
          v66 = v19 - 1;
          if ( (*(_QWORD *)(v20 + 8LL * (v21 >> 6)) & (1LL << v21)) == 0 )
          {
            v60 = v61;
            v62 = v15;
            v22 = sub_16A58F0(v63);
            LOBYTE(v15) = v62;
            v14 = v60;
            if ( v66 == v22 )
            {
LABEL_11:
              v23 = sub_15FF7F0(v17);
              v24 = *(_WORD *)(a2 + 18) & 0x7FFF;
              if ( v23 )
                v25 = sub_15FF470(v24);
              else
                v25 = sub_15FF420(v24);
              v26 = sub_15FF5D0(v25);
              goto LABEL_14;
            }
          }
        }
      }
    }
  }
  if ( v14 == 34 )
  {
    v67 = v15;
    sub_13A38D0((__int64)&v74, a4);
    v47 = v75;
    v48 = v67;
    if ( v75 > 0x40 )
    {
      sub_16A8F40((__int64 *)&v74);
      v47 = v75;
      v49 = (const void *)v74;
      v48 = v67;
    }
    else
    {
      v49 = (const void *)(~v74 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v75));
      v74 = (unsigned __int64)v49;
    }
    v77 = v47;
    v76 = (unsigned __int64)v49;
    v75 = 0;
    if ( *(_DWORD *)(v63 + 8) <= 0x40u )
    {
      if ( *(const void **)v63 != v49 )
      {
LABEL_42:
        if ( v47 > 0x40 && v49 )
        {
          v69 = v48;
          j_j___libc_free_0_0(v49);
          v48 = v69;
        }
        if ( v75 > 0x40 && v74 )
        {
          v70 = v48;
          j_j___libc_free_0_0(v74);
          v48 = v70;
        }
        if ( !v48 )
          return 0;
        v82 = 257;
        v51 = sub_1648A60(56, 2u);
        v32 = v51;
        if ( v51 )
          sub_17582E0((__int64)v51, 36, (__int64)v8, (__int64)v7, (__int64)&v80);
        return v32;
      }
    }
    else
    {
      v68 = v48;
      v50 = sub_16A5220(v63, (const void **)&v76);
      v48 = v68;
      if ( !v50 )
        goto LABEL_42;
    }
    sub_13A38D0((__int64)&v78, a4);
    sub_16A7490((__int64)&v78, 1);
    v53 = v79;
    v79 = 0;
    v81 = v53;
    v80 = v78;
    v71 = v78;
    v54 = sub_14A9C60((__int64)&v80);
    v48 = v54;
    if ( v53 > 0x40 )
    {
      if ( v71 )
      {
        v55 = v71;
        v72 = v54;
        j_j___libc_free_0_0(v55);
        v48 = v72;
        if ( v79 > 0x40 )
        {
          if ( v78 )
          {
            j_j___libc_free_0_0(v78);
            v48 = v72;
          }
        }
      }
    }
    goto LABEL_42;
  }
  if ( v14 != 36 )
    return 0;
  sub_13A38D0((__int64)&v78, a4);
  if ( v79 > 0x40 )
    sub_16A8F40((__int64 *)&v78);
  else
    v78 = ~v78 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v79);
  sub_16A7400((__int64)&v78);
  v52 = v79;
  v79 = 0;
  v81 = v52;
  v80 = v78;
  if ( !sub_1455820(v63, &v80) )
  {
    sub_135E100((__int64 *)&v80);
    sub_135E100((__int64 *)&v78);
    return 0;
  }
  v56 = sub_14A9C60(a4);
  sub_135E100((__int64 *)&v80);
  sub_135E100((__int64 *)&v78);
  if ( !v56 )
    return 0;
  v82 = 257;
  v57 = sub_1648A60(56, 2u);
  v32 = v57;
  if ( v57 )
    sub_17582E0((__int64)v57, 35, (__int64)v8, (__int64)v7, (__int64)&v80);
  return v32;
}
