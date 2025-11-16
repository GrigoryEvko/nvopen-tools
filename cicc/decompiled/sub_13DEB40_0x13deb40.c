// Function: sub_13DEB40
// Address: 0x13deb40
//
unsigned __int8 *__fastcall sub_13DEB40(_QWORD *a1, __int64 a2, char a3, char a4, __int64 *a5, int a6)
{
  __int64 v6; // r15
  __int64 v11; // rax
  unsigned __int8 *result; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r13d
  __int64 v17; // rdi
  unsigned int v18; // r13d
  int v19; // r13d
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // rdx
  int v22; // eax
  int v23; // eax
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rsi
  __int64 v32; // rdi
  char v33; // al
  int v34; // eax
  unsigned __int8 *v35; // rsi
  int v36; // eax
  int v37; // eax
  unsigned __int8 **v38; // rdx
  unsigned __int8 *v39; // rdx
  __int64 *v40; // rsi
  unsigned __int8 *v41; // r13
  unsigned __int8 *v42; // rax
  unsigned int v43; // r8d
  unsigned __int8 *v44; // r9
  unsigned __int8 *v45; // rax
  unsigned __int8 *v46; // r13
  unsigned __int8 *v47; // rax
  unsigned int v48; // r8d
  unsigned __int8 *v49; // r9
  unsigned __int8 *v50; // rax
  unsigned __int8 *v51; // rdx
  unsigned __int8 *v52; // r13
  unsigned __int8 *v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned __int8 *v57; // [rsp+0h] [rbp-70h]
  unsigned __int8 *v58; // [rsp+0h] [rbp-70h]
  unsigned int v60; // [rsp+8h] [rbp-68h]
  unsigned int v61; // [rsp+8h] [rbp-68h]
  unsigned __int64 v62; // [rsp+18h] [rbp-58h] BYREF
  __int64 v63; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v64; // [rsp+28h] [rbp-48h]
  __int64 v65[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = a2;
  LOBYTE(v11) = *((_BYTE *)a1 + 16);
  if ( (unsigned __int8)v11 <= 0x10u )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      result = (unsigned __int8 *)sub_14D6F90(13, a1, a2, *a5);
      if ( result )
        return result;
      v11 = *((unsigned __int8 *)a1 + 16);
    }
    if ( (_BYTE)v11 == 9 )
      return (unsigned __int8 *)sub_1599EF0(*a1);
  }
  if ( *(_BYTE *)(a2 + 16) == 9 )
    return (unsigned __int8 *)sub_1599EF0(*a1);
  if ( sub_13CD190(a2) )
    return (unsigned __int8 *)a1;
  if ( a1 == (_QWORD *)a2 )
    return (unsigned __int8 *)sub_15A06D0(*a1);
  if ( !sub_13CD190((__int64)a1) )
    goto LABEL_12;
  if ( a4 )
    return (unsigned __int8 *)sub_15A06D0(*a1);
  sub_14C2530((unsigned int)&v63, a2, *a5, 0, a5[3], a5[4], a5[2], 0);
  if ( v64 > 0x40 )
  {
    v18 = v64 - 1;
    if ( sub_13D0200(&v63, v64 - 1) || v18 != (unsigned int)sub_16A58F0(&v63) )
      goto LABEL_26;
LABEL_78:
    if ( a3 )
      v6 = sub_15A06D0(*a1);
    sub_135E100(v65);
    sub_135E100(&v63);
    return (unsigned __int8 *)v6;
  }
  if ( v63 == (1LL << ((unsigned __int8)v64 - 1)) - 1 )
    goto LABEL_78;
LABEL_26:
  sub_135E100(v65);
  sub_135E100(&v63);
LABEL_12:
  v16 = *((unsigned __int8 *)a1 + 16);
  if ( !a6 )
    goto LABEL_13;
  if ( (_BYTE)v16 == 35 )
  {
    v15 = *(a1 - 6);
    if ( !v15 )
      goto LABEL_46;
    v46 = (unsigned __int8 *)*(a1 - 3);
    if ( !v46 )
      goto LABEL_46;
LABEL_86:
    v58 = (unsigned __int8 *)v15;
    v47 = (unsigned __int8 *)sub_13DDBD0(13, v46, (unsigned __int8 *)a2, a5, a6 - 1);
    v48 = a6 - 1;
    v49 = v58;
    if ( v47 )
    {
      result = (unsigned __int8 *)sub_13DDBD0(11, v58, v47, a5, v48);
      if ( result )
        return result;
      v49 = v58;
      v48 = a6 - 1;
    }
    v61 = v48;
    v50 = (unsigned __int8 *)sub_13DDBD0(13, v49, (unsigned __int8 *)a2, a5, v48);
    v14 = v61;
    if ( v50 )
    {
      result = (unsigned __int8 *)sub_13DDBD0(11, v46, v50, a5, v61);
      if ( result )
        return result;
    }
    goto LABEL_46;
  }
  if ( (_BYTE)v16 == 5 && *((_WORD *)a1 + 9) == 11 )
  {
    v54 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
    v13 = 4 * v54;
    v15 = a1[-3 * v54];
    if ( v15 )
    {
      v46 = (unsigned __int8 *)a1[3 * (1 - v54)];
      if ( v46 )
        goto LABEL_86;
    }
  }
LABEL_46:
  v33 = *(_BYTE *)(a2 + 16);
  if ( v33 == 35 )
  {
    v41 = *(unsigned __int8 **)(a2 - 48);
    if ( !v41 )
      goto LABEL_50;
    v15 = *(_QWORD *)(a2 - 24);
    if ( !v15 )
      goto LABEL_50;
  }
  else
  {
    if ( v33 != 5 )
    {
LABEL_74:
      if ( v33 == 37 )
      {
        v51 = *(unsigned __int8 **)(a2 - 48);
        if ( !v51 )
          goto LABEL_50;
        v52 = *(unsigned __int8 **)(a2 - 24);
        if ( !v52 )
          goto LABEL_50;
        goto LABEL_94;
      }
      if ( v33 != 5 )
        goto LABEL_50;
LABEL_49:
      if ( *(_WORD *)(a2 + 18) != 13 )
        goto LABEL_50;
      v56 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v13 = 4 * v56;
      v51 = *(unsigned __int8 **)(a2 - 24 * v56);
      if ( !v51 )
        goto LABEL_50;
      v13 = 1 - v56;
      v52 = *(unsigned __int8 **)(a2 + 24 * (1 - v56));
      if ( !v52 )
        goto LABEL_50;
LABEL_94:
      v53 = (unsigned __int8 *)sub_13DDBD0(13, (unsigned __int8 *)a1, v51, a5, a6 - 1);
      if ( v53 )
      {
        result = (unsigned __int8 *)sub_13DDBD0(11, v53, v52, a5, a6 - 1);
        if ( result )
          return result;
      }
LABEL_50:
      v16 = *((unsigned __int8 *)a1 + 16);
      if ( (unsigned __int8)v16 > 0x17u )
      {
        v34 = (unsigned __int8)v16 - 24;
      }
      else
      {
        if ( (_BYTE)v16 != 5 )
          goto LABEL_13;
        v34 = *((unsigned __int16 *)a1 + 9);
      }
      if ( v34 == 36 )
      {
        v35 = *(unsigned __int8 **)sub_13CF970((__int64)a1);
        if ( v35 )
        {
          v36 = *(unsigned __int8 *)(v6 + 16);
          if ( (unsigned __int8)v36 > 0x17u )
          {
            v37 = v36 - 24;
          }
          else
          {
            if ( (_BYTE)v36 != 5 )
              goto LABEL_13;
            v37 = *(unsigned __int16 *)(v6 + 18);
          }
          if ( v37 == 36 )
          {
            v38 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
                ? *(unsigned __int8 ***)(v6 - 8)
                : (unsigned __int8 **)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
            v39 = *v38;
            if ( v39 )
            {
              if ( *(_QWORD *)v39 == *(_QWORD *)v35 )
              {
                v40 = sub_13DDBD0(13, v35, v39, a5, a6 - 1);
                if ( v40 )
                {
                  result = (unsigned __int8 *)sub_13CBA60(36, v40, *a1, a5, v14);
                  if ( result )
                    return result;
                }
                v16 = *((unsigned __int8 *)a1 + 16);
              }
            }
          }
        }
      }
LABEL_13:
      if ( (unsigned __int8)v16 > 0x17u )
      {
        v19 = v16 - 24;
      }
      else
      {
        if ( (_BYTE)v16 != 5 )
          goto LABEL_15;
        v19 = *((unsigned __int16 *)a1 + 9);
      }
      if ( v19 == 45 )
      {
        v20 = (*((_BYTE *)a1 + 23) & 0x40) != 0
            ? (unsigned __int64 *)*(a1 - 1)
            : &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
        v21 = *v20;
        if ( *v20 )
        {
          v22 = *(unsigned __int8 *)(v6 + 16);
          if ( (unsigned __int8)v22 > 0x17u )
          {
            v23 = v22 - 24;
          }
          else
          {
            if ( (_BYTE)v22 != 5 )
              goto LABEL_15;
            v23 = *(unsigned __int16 *)(v6 + 18);
          }
          if ( v23 == 45 )
          {
            if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
            {
              v24 = *(__int64 **)(v6 - 8);
            }
            else
            {
              v13 = 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
              v24 = (__int64 *)(v6 - v13);
            }
            v25 = *v24;
            if ( v25 )
            {
              v26 = *a5;
              v62 = v21;
              v63 = v25;
              v27 = sub_13CCBD0(v26, &v62, 0, v13, v14, v15);
              v31 = sub_13CCBD0(v26, (unsigned __int64 *)&v63, 0, v28, v29, v30);
              if ( v62 == v63 )
              {
                v32 = sub_15A2B60(v27, v31, 0, 0);
                if ( v32 )
                  return (unsigned __int8 *)sub_15A4750(v32, *a1, 1);
              }
            }
          }
        }
      }
LABEL_15:
      if ( !a6 )
        return 0;
      v17 = *a1;
      if ( *(_BYTE *)(*a1 + 8LL) == 16 )
        v17 = **(_QWORD **)(v17 + 16);
      if ( (unsigned __int8)sub_1642F90(v17, 1) )
        return sub_13DE280((unsigned __int8 *)a1, (unsigned __int8 *)v6, a5, a6 - 1);
      else
        return 0;
    }
    if ( *(_WORD *)(a2 + 18) != 11 )
      goto LABEL_49;
    v55 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v13 = 4 * v55;
    v41 = *(unsigned __int8 **)(a2 - 24 * v55);
    if ( !v41 )
      goto LABEL_49;
    v15 = *(_QWORD *)(a2 + 24 * (1 - v55));
    if ( !v15 )
      goto LABEL_49;
  }
  v57 = (unsigned __int8 *)v15;
  v42 = (unsigned __int8 *)sub_13DDBD0(13, (unsigned __int8 *)a1, v41, a5, a6 - 1);
  v43 = a6 - 1;
  v44 = v57;
  if ( v42 )
  {
    result = (unsigned __int8 *)sub_13DDBD0(13, v42, v57, a5, v43);
    if ( result )
      return result;
    v44 = v57;
    v43 = a6 - 1;
  }
  v60 = v43;
  v45 = (unsigned __int8 *)sub_13DDBD0(13, (unsigned __int8 *)a1, v44, a5, v43);
  v14 = v60;
  if ( !v45 || (result = (unsigned __int8 *)sub_13DDBD0(13, v45, v41, a5, v60)) == 0 )
  {
    v33 = *(_BYTE *)(a2 + 16);
    goto LABEL_74;
  }
  return result;
}
