// Function: sub_14D7A60
// Address: 0x14d7a60
//
__int64 __fastcall sub_14D7A60(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r15
  bool v11; // zf
  __int16 v12; // ax
  _QWORD *v13; // r13
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rax
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // r15
  unsigned __int64 v27; // r12
  _BYTE *v28; // rbx
  __int64 v29; // r8
  __int16 v30; // ax
  __int64 v31; // rax
  _QWORD *v32; // r13
  unsigned int v33; // ebx
  unsigned int v34; // eax
  unsigned int v35; // r8d
  __int64 v36; // rsi
  __int64 v37; // rax
  _QWORD *v38; // r9
  __int64 v39; // rsi
  _QWORD *v40; // rcx
  _QWORD *v41; // rax
  unsigned __int64 v42; // rdx
  int v43; // eax
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r12
  _QWORD *v47; // [rsp+8h] [rbp-128h]
  _BOOL4 v48; // [rsp+10h] [rbp-120h]
  _QWORD *v49; // [rsp+10h] [rbp-120h]
  int v50; // [rsp+18h] [rbp-118h]
  __int64 v51; // [rsp+18h] [rbp-118h]
  int v52; // [rsp+20h] [rbp-110h]
  __int64 v53; // [rsp+38h] [rbp-F8h]
  unsigned int v54; // [rsp+48h] [rbp-E8h]
  unsigned int v55; // [rsp+48h] [rbp-E8h]
  __int64 v56; // [rsp+48h] [rbp-E8h]
  _BYTE v57[8]; // [rsp+58h] [rbp-D8h] BYREF
  _BYTE *v58; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v59; // [rsp+68h] [rbp-C8h]
  _BYTE v60[64]; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v61; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-78h]
  _BYTE v63[112]; // [rsp+C0h] [rbp-70h] BYREF

  v5 = a3;
  v6 = a2;
  switch ( (int)a1 )
  {
    case '$':
    case '%':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '0':
      if ( (_DWORD)a1 != 48 )
        goto LABEL_3;
      v8 = a3;
      if ( *(_BYTE *)(a3 + 8) == 16 )
        v8 = **(_QWORD **)(a3 + 16);
      v9 = *(_DWORD *)(v8 + 8);
      v10 = a2;
      v58 = v60;
      v11 = *(_BYTE *)(a2 + 16) == 5;
      v54 = v9 >> 8;
      v59 = 0x800000000LL;
      if ( !v11 )
        goto LABEL_3;
      break;
    case '-':
      if ( *(_BYTE *)(a2 + 16) == 5 && *(_WORD *)(a2 + 18) == 46 )
      {
        v32 = *(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v33 = sub_16431D0(*v32);
        v34 = sub_15A9570(a4, *(_QWORD *)a2);
        v35 = v34;
        if ( v33 > v34 )
        {
          LODWORD(v62) = v33;
          if ( v33 > 0x40 )
          {
            v55 = v34;
            sub_16A4EF0(&v61, 0, 0);
            v35 = v55;
          }
          else
          {
            v61 = 0;
          }
          if ( v35 )
          {
            if ( v35 > 0x40 )
            {
              sub_16A5260(&v61, 0, v35);
            }
            else
            {
              v44 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v35);
              if ( (unsigned int)v62 > 0x40 )
                *(_QWORD *)v61 |= v44;
              else
                v61 |= v44;
            }
          }
          v45 = sub_16498A0(a2);
          v46 = sub_159C0E0(v45, &v61);
          if ( (unsigned int)v62 > 0x40 && v61 )
            j_j___libc_free_0_0(v61);
          v32 = (_QWORD *)sub_15A2CF0(v32, v46);
        }
        return sub_15A4750(v32, v5, 0);
      }
      else
      {
        a1 = 45;
        return sub_15A46C0(a1, a2, a3, 0);
      }
    case '.':
      if ( *(_BYTE *)(a2 + 16) != 5 )
        return sub_15A46C0(46, a2, v5, 0);
      if ( *(_WORD *)(a2 + 18) != 45 )
        return sub_15A46C0(46, a2, v5, 0);
      v13 = *(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v14 = sub_15A9570(a4, *v13);
      if ( v14 > (unsigned int)sub_16431D0(*(_QWORD *)a2) )
        return sub_15A46C0(46, a2, v5, 0);
      v15 = *v13;
      if ( *(_BYTE *)(*v13 + 8LL) == 16 )
        v15 = **(_QWORD **)(v15 + 16);
      v16 = v5;
      v17 = *(_DWORD *)(v15 + 8) >> 8;
      if ( *(_BYTE *)(v5 + 8) == 16 )
        v16 = **(_QWORD **)(v5 + 16);
      if ( *(_DWORD *)(v16 + 8) >> 8 != v17 )
        return sub_15A46C0(46, a2, v5, 0);
      v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( (unsigned __int8)sub_1593BB0(v6) )
      {
LABEL_6:
        if ( *(_BYTE *)(v5 + 8) != 9 )
          return sub_15A06D0(v5);
      }
      return sub_14D44C0(v6, v5, a4);
    case '/':
      if ( (unsigned __int8)sub_1593BB0(a2) )
        goto LABEL_6;
      return sub_14D44C0(v6, v5, a4);
  }
  while ( 1 )
  {
    v12 = *(_WORD *)(v10 + 18);
    if ( v12 != 47 )
      break;
    v18 = (unsigned int)v59;
    if ( (unsigned int)v59 >= HIDWORD(v59) )
    {
      sub_16CD150(&v58, v60, 0, 8);
      v18 = (unsigned int)v59;
    }
    *(_QWORD *)&v58[8 * v18] = v10;
    v19 = *(_DWORD *)(v10 + 20);
    LODWORD(v59) = v59 + 1;
    v10 = *(_QWORD *)(v10 - 24LL * (v19 & 0xFFFFFFF));
    if ( !v10 )
      BUG();
LABEL_33:
    if ( *(_BYTE *)(v10 + 16) != 5 )
      goto LABEL_20;
  }
  if ( v12 != 48 )
  {
    if ( v12 != 32 )
      goto LABEL_20;
    v20 = (unsigned int)v59;
    if ( (unsigned int)v59 >= HIDWORD(v59) )
    {
      sub_16CD150(&v58, v60, 0, 8);
      v20 = (unsigned int)v59;
    }
    *(_QWORD *)&v58[8 * v20] = v10;
    LODWORD(v59) = v59 + 1;
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      v21 = *(__int64 **)(v10 - 8);
    else
      v21 = (__int64 *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    v10 = *v21;
    goto LABEL_33;
  }
  v22 = **(_QWORD **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v22 + 8) == 16 )
    v22 = **(_QWORD **)(v22 + 16);
  if ( v54 != *(_DWORD *)(v22 + 8) >> 8 )
    goto LABEL_20;
  v23 = (unsigned int)v59;
  if ( (unsigned int)v59 >= HIDWORD(v59) )
  {
    sub_16CD150(&v58, v60, 0, 8);
    v23 = (unsigned int)v59;
  }
  *(_QWORD *)&v58[8 * v23] = v10;
  v24 = (unsigned int)(v59 + 1);
  v25 = *(_DWORD *)(v10 + 20);
  LODWORD(v59) = v59 + 1;
  v26 = *(_QWORD *)(v10 - 24LL * (v25 & 0xFFFFFFF));
  if ( !v26 )
    goto LABEL_20;
  if ( v58 == &v58[8 * v24] )
    goto LABEL_53;
  v53 = a2;
  v27 = (unsigned __int64)v58;
  v28 = &v58[8 * v24];
  while ( 2 )
  {
    v29 = *((_QWORD *)v28 - 1);
    v30 = *(_WORD *)(v29 + 18);
    if ( v30 != 32 )
    {
      if ( (unsigned __int16)(v30 - 47) <= 1u )
      {
        v31 = sub_1646BA0(*(_QWORD *)(*(_QWORD *)v29 + 24LL), v54);
        v26 = sub_15A4510(v26, v31, 0);
        goto LABEL_51;
      }
      v6 = v53;
LABEL_20:
      if ( v58 != v60 )
        _libc_free((unsigned __int64)v58);
LABEL_3:
      a3 = v5;
      a2 = v6;
      a1 = (unsigned int)a1;
      return sub_15A46C0(a1, a2, a3, 0);
    }
    v36 = 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v29 + 23) & 0x40) != 0 )
    {
      v37 = *(_QWORD *)(v29 - 8);
      v38 = (_QWORD *)(v37 + v36);
    }
    else
    {
      v38 = (_QWORD *)*((_QWORD *)v28 - 1);
      v37 = v29 - v36;
    }
    v39 = v36 - 24;
    v40 = v63;
    v41 = (_QWORD *)(v37 + 24);
    v62 = 0x800000000LL;
    v61 = (unsigned __int64)v63;
    v42 = 0xAAAAAAAAAAAAAAABLL * (v39 >> 3);
    if ( (unsigned __int64)v39 > 0xC0 )
    {
      v47 = v41;
      v49 = v38;
      v51 = v29;
      sub_16CD150(&v61, v63, v42, 8);
      v41 = v47;
      v38 = v49;
      v29 = v51;
      LODWORD(v42) = -1431655765 * (v39 >> 3);
      v40 = (_QWORD *)(v61 + 8LL * (unsigned int)v62);
    }
    for ( ; v41 != v38; ++v40 )
    {
      if ( v40 )
        *v40 = *v41;
      v41 += 3;
    }
    v57[4] = 0;
    LODWORD(v62) = v42 + v62;
    v50 = v61;
    v52 = v62;
    v48 = (*(_BYTE *)(v29 + 17) & 2) != 0;
    v43 = sub_16348C0(v29);
    v26 = sub_15A2E80(v43, v26, v50, v52, v48, (unsigned int)v57, 0);
    if ( (_BYTE *)v61 != v63 )
      _libc_free(v61);
LABEL_51:
    v28 -= 8;
    if ( (_BYTE *)v27 != v28 )
      continue;
    break;
  }
  v6 = v53;
LABEL_53:
  result = sub_15A4510(v26, v5, 0);
  if ( v58 != v60 )
  {
    v56 = result;
    _libc_free((unsigned __int64)v58);
    result = v56;
  }
  if ( !result )
    goto LABEL_3;
  return result;
}
