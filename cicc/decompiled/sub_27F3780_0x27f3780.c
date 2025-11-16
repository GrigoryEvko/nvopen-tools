// Function: sub_27F3780
// Address: 0x27f3780
//
void __fastcall sub_27F3780(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // r15
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // r13
  _QWORD *v9; // rdx
  _QWORD *v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned int i; // r14d
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 *v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // r15
  unsigned int v23; // r14d
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rax
  __int64 *v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 *v34; // rdx
  __int64 *v35; // r13
  __int64 *v36; // r14
  __int64 v37; // rsi
  __int64 *v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // r12
  __int64 v44; // rsi
  __int64 *v45; // rax
  __int64 *v46; // rdx
  __int64 *v47; // r14
  __int64 v48; // rsi
  __int64 *v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 *v53; // rdx
  __int64 v54; // [rsp+8h] [rbp-C8h]
  int v55; // [rsp+8h] [rbp-C8h]
  int v56; // [rsp+14h] [rbp-BCh]
  char v57; // [rsp+14h] [rbp-BCh]
  __int64 v58; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v59; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v60; // [rsp+28h] [rbp-A8h]
  __int64 v61; // [rsp+30h] [rbp-A0h]
  int v62; // [rsp+38h] [rbp-98h]
  char v63; // [rsp+3Ch] [rbp-94h]
  char v64; // [rsp+40h] [rbp-90h] BYREF
  __int64 v65; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v66; // [rsp+68h] [rbp-68h]
  __int64 v67; // [rsp+70h] [rbp-60h]
  int v68; // [rsp+78h] [rbp-58h]
  char v69; // [rsp+7Ch] [rbp-54h]
  char v70; // [rsp+80h] [rbp-50h] BYREF

  v58 = a2;
  if ( !(_BYTE)qword_4FFE5C8 )
    return;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    return;
  v5 = sub_D484B0(*(_QWORD *)(a1 + 16), a2, a3, a4);
  if ( !v5 )
    return;
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(v58 - 32);
  v8 = *(_QWORD *)(v58 - 64);
  if ( *(_BYTE *)(v6 + 84) )
  {
    v9 = *(_QWORD **)(v6 + 64);
    v10 = &v9[*(unsigned int *)(v6 + 76)];
    if ( v9 == v10 )
      return;
    v11 = *(_QWORD **)(v6 + 64);
    while ( v7 != *v11 )
    {
      if ( v10 == ++v11 )
        return;
    }
  }
  else
  {
    if ( !sub_C8CA60(v6 + 56, *(_QWORD *)(v58 - 32)) )
      return;
    v32 = *(_QWORD *)(a1 + 16);
    if ( !*(_BYTE *)(v32 + 84) )
    {
      if ( !sub_C8CA60(v32 + 56, v8) )
        return;
      goto LABEL_13;
    }
    v9 = *(_QWORD **)(v32 + 64);
    v10 = &v9[*(unsigned int *)(v32 + 76)];
    if ( v10 == v9 )
      return;
  }
  while ( v8 != *v9 )
  {
    if ( ++v9 == v10 )
      return;
  }
LABEL_13:
  if ( v8 == v7 )
    return;
  v59 = 0;
  v60 = (__int64 *)&v64;
  v66 = (__int64 *)&v70;
  v12 = *(_QWORD *)(v7 + 48);
  v61 = 4;
  v13 = v12 & 0xFFFFFFFFFFFFFFF8LL;
  v63 = 1;
  v62 = 0;
  v65 = 0;
  v67 = 4;
  v68 = 0;
  v69 = 1;
  if ( v13 == v7 + 48 )
    goto LABEL_93;
  if ( !v13 )
    goto LABEL_64;
  v54 = v13 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA || (v56 = sub_B46E30(v13 - 24)) == 0 )
  {
LABEL_93:
    v21 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v21 == v8 + 48 )
      goto LABEL_37;
LABEL_26:
    if ( v21 )
    {
      v22 = v21 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v21 - 24) - 30 > 0xA )
        goto LABEL_36;
      v55 = sub_B46E30(v22);
      if ( !v55 )
        goto LABEL_36;
      v23 = 0;
      v57 = v69;
      while ( 1 )
      {
        v26 = sub_B46EC0(v22, v23);
        if ( !v57 )
          goto LABEL_47;
        v29 = v66;
        v24 = &v66[HIDWORD(v67)];
        if ( v66 != v24 )
        {
          while ( v26 != *v29 )
          {
            if ( v24 == ++v29 )
              goto LABEL_51;
          }
          goto LABEL_35;
        }
LABEL_51:
        if ( HIDWORD(v67) < (unsigned int)v67 )
        {
          ++HIDWORD(v67);
          *v24 = v26;
          ++v65;
          v57 = v69;
        }
        else
        {
LABEL_47:
          sub_C8CC70((__int64)&v65, v26, (__int64)v24, v25, v27, v28);
          v57 = v69;
        }
LABEL_35:
        if ( v55 == ++v23 )
          goto LABEL_36;
      }
    }
LABEL_64:
    BUG();
  }
  for ( i = 0; i != v56; ++i )
  {
    v17 = sub_B46EC0(v54, i);
    if ( !v5 )
    {
LABEL_48:
      sub_C8CC70((__int64)&v59, v17, (__int64)v15, v16, v18, v19);
      v5 = v63;
      continue;
    }
    v20 = v60;
    v15 = &v60[HIDWORD(v61)];
    if ( v60 == v15 )
    {
LABEL_49:
      if ( HIDWORD(v61) >= (unsigned int)v61 )
        goto LABEL_48;
      ++HIDWORD(v61);
      *v15 = v17;
      v5 = v63;
      ++v59;
    }
    else
    {
      while ( v17 != *v20 )
      {
        if ( v15 == ++v20 )
          goto LABEL_49;
      }
    }
  }
  v21 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v21 != v8 + 48 )
    goto LABEL_26;
LABEL_36:
  if ( !v63 )
  {
    if ( !sub_C8CA60((__int64)&v59, v8) )
      goto LABEL_58;
LABEL_41:
    if ( (unsigned __int8)sub_B19D00(*(_QWORD *)(a1 + 8), v58, v8) )
      *sub_27F3530(a1 + 64, &v58) = v8;
    goto LABEL_43;
  }
LABEL_37:
  v30 = v60;
  v31 = &v60[HIDWORD(v61)];
  if ( v60 != v31 )
  {
    while ( v8 != *v30 )
    {
      if ( v31 == ++v30 )
        goto LABEL_58;
    }
    goto LABEL_41;
  }
LABEL_58:
  if ( v69 )
  {
    v33 = v66;
    v34 = &v66[HIDWORD(v67)];
    if ( v66 != v34 )
    {
      do
      {
        v8 = *v33;
        if ( v7 == *v33 )
          goto LABEL_41;
        ++v33;
      }
      while ( v34 != v33 );
    }
  }
  else if ( sub_C8CA60((__int64)&v65, v7) )
  {
    v8 = v7;
    goto LABEL_41;
  }
  v35 = v60;
  if ( v63 )
  {
    v47 = &v60[HIDWORD(v61)];
    if ( v47 == v60 )
      goto LABEL_74;
    while ( 1 )
    {
      v48 = *v35;
      if ( v69 )
      {
        v49 = v66;
        v50 = &v66[HIDWORD(v67)];
        if ( v66 == v50 )
          goto LABEL_107;
        while ( v48 != *v49 )
        {
          if ( v50 == ++v49 )
            goto LABEL_107;
        }
      }
      else if ( !sub_C8CA60((__int64)&v65, v48) )
      {
LABEL_107:
        v51 = *--v47;
        *v35 = v51;
        --HIDWORD(v61);
        ++v59;
        goto LABEL_104;
      }
      ++v35;
LABEL_104:
      if ( v35 == v47 )
        goto LABEL_74;
    }
  }
  v36 = &v60[(unsigned int)v61];
  if ( v36 != v60 )
  {
    do
    {
      v37 = *v35;
      if ( (unsigned __int64)*v35 < 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v69 )
        {
          v38 = v66;
          v39 = &v66[HIDWORD(v67)];
          if ( v66 != v39 )
          {
            while ( v37 != *v38 )
            {
              if ( v39 == ++v38 )
                goto LABEL_90;
            }
            goto LABEL_73;
          }
LABEL_90:
          *v35 = -2;
          ++v62;
          ++v59;
          goto LABEL_73;
        }
        if ( !sub_C8CA60((__int64)&v65, v37) )
          goto LABEL_90;
      }
LABEL_73:
      ++v35;
    }
    while ( v35 != v36 );
  }
LABEL_74:
  v40 = HIDWORD(v61);
  if ( HIDWORD(v61) - v62 == 1 )
  {
    v52 = v60;
    if ( !v63 )
      v40 = (unsigned int)v61;
    v53 = &v60[v40];
    if ( v60 != v53 )
    {
      while ( (unsigned __int64)*v52 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( ++v52 == v53 )
          goto LABEL_113;
      }
      v53 = v52;
    }
LABEL_113:
    v8 = *v53;
    if ( *v53 )
      goto LABEL_41;
    goto LABEL_43;
  }
  if ( HIDWORD(v61) == v62 )
    goto LABEL_43;
  v41 = *(_QWORD *)(v7 + 72);
  v42 = *(_QWORD *)(v41 + 80);
  v43 = v41 + 72;
  if ( v41 + 72 == v42 )
    goto LABEL_85;
  while ( 2 )
  {
    v44 = v42 - 24;
    if ( !v42 )
      v44 = 0;
    if ( !v63 )
    {
      if ( sub_C8CA60((__int64)&v59, v44) )
        goto LABEL_84;
      goto LABEL_87;
    }
    v45 = v60;
    v46 = &v60[HIDWORD(v61)];
    if ( v60 == v46 )
    {
LABEL_87:
      v42 = *(_QWORD *)(v42 + 8);
      if ( v43 == v42 )
        goto LABEL_85;
      continue;
    }
    break;
  }
  while ( v44 != *v45 )
  {
    if ( v46 == ++v45 )
      goto LABEL_87;
  }
LABEL_84:
  if ( v42 )
  {
LABEL_85:
    v8 = v42 - 24;
    goto LABEL_41;
  }
LABEL_43:
  if ( !v69 )
    _libc_free((unsigned __int64)v66);
  if ( !v63 )
    _libc_free((unsigned __int64)v60);
}
