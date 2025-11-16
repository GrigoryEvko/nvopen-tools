// Function: sub_1760CE0
// Address: 0x1760ce0
//
__int64 __fastcall sub_1760CE0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r12
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // r15
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 *v27; // r13
  __int64 v28; // r14
  _QWORD *v29; // rax
  _QWORD *v30; // r13
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 **v34; // rdi
  __int64 ****v35; // r13
  __int64 ***v36; // r13
  __int16 v37; // r14
  __int64 v38; // r15
  __int16 v39; // r14
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int16 v43; // r14
  __int64 ****v44; // r13
  __int64 ***v45; // r15
  __int64 v46; // rax
  __int64 v47; // r13
  _QWORD *v48; // rax
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // r15
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // r14
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // [rsp+8h] [rbp-68h]
  const char *v60; // [rsp+10h] [rbp-60h] BYREF
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v63; // [rsp+30h] [rbp-40h]

  v12 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v12 + 16) > 0x10u )
    return 0;
  v13 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v13 + 16) <= 0x17u )
    return 0;
  switch ( *(_BYTE *)(v13 + 16) )
  {
    case '6':
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
        v16 = *(_QWORD *)(v13 - 8);
      else
        v16 = v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      v17 = *(_QWORD *)v16;
      if ( *(_BYTE *)(*(_QWORD *)v16 + 16LL) == 56 )
      {
        v58 = *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v58 + 16) == 3
          && (*(_BYTE *)(v58 + 80) & 1) != 0
          && !sub_15E4F60(*(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF))) )
        {
          __asm { jmp     rax }
        }
      }
      return 0;
    case '8':
      if ( !sub_1593BB0(*(_QWORD *)(a2 - 24), a2, a3, a4) || !(unsigned __int8)sub_15FA1F0(v13) )
        return 0;
      v43 = *(_WORD *)(a2 + 18) & 0x7FFF;
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
        v44 = *(__int64 *****)(v13 - 8);
      else
        v44 = (__int64 ****)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v45 = *v44;
      v46 = sub_15A06D0(**v44, a2, v41, v42);
      v63 = 257;
      v47 = v46;
      v48 = sub_1648A60(56, 2u);
      v18 = (__int64)v48;
      if ( v48 )
        sub_17582E0((__int64)v48, v43, (__int64)v45, v47, (__int64)v62);
      return v18;
    case 'F':
      if ( !sub_1593BB0(*(_QWORD *)(a2 - 24), a2, a3, a4) )
        return 0;
      v31 = *(_QWORD *)v12;
      v34 = (__int64 **)sub_15A9650(a1[333], *(_QWORD *)v12);
      v35 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
          ? *(__int64 *****)(v13 - 8)
          : (__int64 ****)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v36 = *v35;
      if ( v34 != *v36 )
        return 0;
      v37 = *(_WORD *)(a2 + 18);
      v38 = sub_15A06D0(v34, v31, v32, v33);
      v39 = v37 & 0x7FFF;
      v63 = 257;
      v40 = sub_1648A60(56, 2u);
      v18 = (__int64)v40;
      if ( v40 )
        sub_17582E0((__int64)v40, v39, (__int64)v36, v38, (__int64)v62);
      return v18;
    case 'M':
      if ( *(_QWORD *)(v13 + 40) != *(_QWORD *)(a2 + 40) )
        return 0;
      return sub_17127D0(a1, a2, *(_QWORD *)(a2 - 48), a5, a6, a7, a8, a9, a10, a11, a12);
    case 'O':
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      {
        v20 = *(_QWORD *)(v13 - 8);
        v21 = *(_QWORD *)(v20 + 24);
        if ( *(_BYTE *)(v21 + 16) <= 0x10u )
          goto LABEL_11;
        v50 = *(_QWORD *)(v20 + 48);
        if ( *(_BYTE *)(v50 + 16) > 0x10u )
          return 0;
LABEL_37:
        v59 = sub_15A35F0(*(_WORD *)(a2 + 18) & 0x7FFF, (_QWORD *)v50, (_QWORD *)v12, 0);
        if ( *(_BYTE *)(v59 + 16) == 13 )
        {
          v51 = *(_QWORD *)(v13 + 8);
          v23 = v59;
          v24 = 0;
          if ( !v51 )
            goto LABEL_56;
        }
        else
        {
          v51 = *(_QWORD *)(v13 + 8);
          if ( !v51 )
            return 0;
          v23 = 0;
          v24 = 0;
        }
        goto LABEL_39;
      }
      v49 = 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      v21 = *(_QWORD *)(v13 - v49 + 24);
      if ( *(_BYTE *)(v21 + 16) > 0x10u )
      {
        v50 = *(_QWORD *)(v13 - v49 + 48);
        if ( *(_BYTE *)(v50 + 16) > 0x10u )
          return 0;
        goto LABEL_37;
      }
LABEL_11:
      v22 = sub_15A35F0(*(_WORD *)(a2 + 18) & 0x7FFF, (_QWORD *)v21, (_QWORD *)v12, 0);
      v23 = 0;
      v24 = (__int64 *)v22;
      if ( *(_BYTE *)(v22 + 16) == 13 )
        v23 = v22;
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
        v25 = *(_QWORD *)(v13 - 8);
      else
        v25 = v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      v26 = *(_QWORD *)(v25 + 48);
      if ( *(_BYTE *)(v26 + 16) <= 0x10u )
      {
        v59 = sub_15A35F0(*(_WORD *)(a2 + 18) & 0x7FFF, (_QWORD *)v26, (_QWORD *)v12, 0);
        goto LABEL_17;
      }
      v51 = *(_QWORD *)(v13 + 8);
      v59 = 0;
      if ( !v51 )
        goto LABEL_55;
LABEL_39:
      if ( !*(_QWORD *)(v51 + 8) )
      {
        if ( v24 )
          goto LABEL_42;
        goto LABEL_41;
      }
LABEL_55:
      if ( !v23 )
        return 0;
LABEL_56:
      if ( sub_13D01C0(v23 + 24) )
        return 0;
      if ( !v24 )
      {
        if ( !(unsigned __int8)sub_1759270((__int64)a1, v13, a2, 1u) )
          return 0;
LABEL_41:
        v52 = a1[1];
        v60 = sub_1649960(a2);
        v61 = v53;
        v63 = 261;
        v62[0] = (__int64)&v60;
        v54 = sub_13CF970(v13);
        v24 = (__int64 *)sub_17203D0(v52, *(_WORD *)(a2 + 18) & 0x7FFF, *(_QWORD *)(v54 + 24), v12, v62);
        goto LABEL_42;
      }
      if ( !(unsigned __int8)sub_1759270((__int64)a1, v13, a2, 2u) )
        return 0;
LABEL_42:
      if ( !v59 )
      {
        v55 = a1[1];
        v60 = sub_1649960(a2);
        v61 = v56;
        v63 = 261;
        v62[0] = (__int64)&v60;
        v57 = sub_13CF970(v13);
        v59 = (__int64)sub_17203D0(v55, *(_WORD *)(a2 + 18) & 0x7FFF, *(_QWORD *)(v57 + 48), v12, v62);
      }
LABEL_17:
      v63 = 257;
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
        v27 = *(__int64 **)(v13 - 8);
      else
        v27 = (__int64 *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v28 = *v27;
      v29 = sub_1648A60(56, 3u);
      v18 = (__int64)v29;
      if ( v29 )
      {
        v30 = v29 - 9;
        sub_15F1EA0((__int64)v29, *v24, 55, (__int64)(v29 - 9), 3, 0);
        sub_1593B40(v30, v28);
        sub_1593B40((_QWORD *)(v18 - 48), (__int64)v24);
        sub_1593B40((_QWORD *)(v18 - 24), v59);
        sub_164B780(v18, v62);
      }
      return v18;
    default:
      return 0;
  }
}
