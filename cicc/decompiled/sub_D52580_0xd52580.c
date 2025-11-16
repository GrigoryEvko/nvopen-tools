// Function: sub_D52580
// Address: 0xd52580
//
__int64 __fastcall sub_D52580(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r11
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 *v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  char *v37; // rax
  char *v38; // rcx
  __int64 i; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int8 *v43; // rax
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdi
  __int64 v64; // rdi
  __int64 v65; // rdi
  signed __int64 v66; // rdx
  __int64 v67; // [rsp-158h] [rbp-158h]
  __int64 v69; // [rsp-148h] [rbp-148h]
  __int64 *v70; // [rsp-140h] [rbp-140h]
  __int64 *v71; // [rsp-138h] [rbp-138h]
  __int64 v72; // [rsp-130h] [rbp-130h]
  __int64 v73; // [rsp-128h] [rbp-128h]
  __int64 v74; // [rsp-120h] [rbp-120h]
  __int64 v76; // [rsp-118h] [rbp-118h]
  __int64 v77; // [rsp-118h] [rbp-118h]
  __int64 v78; // [rsp-118h] [rbp-118h]
  __int64 v79; // [rsp-110h] [rbp-110h]
  __int64 v80; // [rsp-110h] [rbp-110h]
  unsigned __int8 *v81; // [rsp-108h] [rbp-108h] BYREF
  unsigned __int8 *v82[17]; // [rsp-100h] [rbp-100h] BYREF
  _BYTE v83[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( *(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8) != 8 )
    return 2;
  v3 = a1;
  v4 = (__int64)a2;
  if ( a1 != *a2 )
    return 2;
  if ( !(unsigned __int8)sub_D4B3D0(a1) )
    return 2;
  if ( !(unsigned __int8)sub_D4B3D0((__int64)a2) )
    return 2;
  v76 = **(_QWORD **)(a1 + 32);
  v7 = sub_D47930(a1);
  v79 = sub_D4B130((__int64)a2);
  v8 = sub_D47930((__int64)a2);
  v74 = sub_D47470((__int64)a2);
  if ( v7 != sub_D46F00(a1) || v8 != sub_D46F00((__int64)a2) || !v74 )
    return 2;
  if ( v76 == v79 )
    goto LABEL_53;
  v9 = sub_D52390(v76, v79, 0);
  if ( v79 == v9 )
    goto LABEL_53;
  v10 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == v9 + 48 )
    goto LABEL_93;
  if ( !v10 )
    BUG();
  v11 = v10 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
LABEL_93:
    BUG();
  if ( *(_BYTE *)(v10 - 24) != 31 || v11 != sub_D4B890((__int64)a2) )
    return 2;
  v12 = sub_AA5930(v74);
  v73 = v13;
  v14 = v12;
  while ( v13 != v14 )
  {
    if ( (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) == 1 )
      break;
    v15 = *(_QWORD *)(v14 + 32);
    if ( !v15 )
      BUG();
    v14 = 0;
    if ( *(_BYTE *)(v15 - 24) == 84 )
      v14 = v15 - 24;
  }
  v69 = v14;
  v16 = sub_D52340(v11);
  v17 = v74;
  v70 = v18;
  v19 = v76;
  v71 = (__int64 *)v16;
  v72 = 0;
  v77 = a3;
  while ( 1 )
  {
    if ( v70 == v71 )
    {
      v3 = a1;
      v4 = (__int64)a2;
      a3 = v77;
      if ( v72 )
      {
        v41 = sub_D47470((__int64)a2);
        if ( v72 == sub_D52390(v41, v72, 0) )
        {
LABEL_54:
          sub_D4B680((__int64)v83, v3, a3);
          result = 3;
          if ( v83[48] )
          {
            v81 = sub_D522E0(v3);
            v43 = (unsigned __int8 *)sub_D4B890(v4);
            if ( v43 )
            {
              v43 = (unsigned __int8 *)*((_QWORD *)v43 - 12);
              if ( (unsigned __int8)(*v43 - 82) >= 2u )
                v43 = 0;
            }
            v82[0] = v43;
            v44 = **(_QWORD **)(v3 + 32);
            v78 = sub_D47930(v3);
            v45 = sub_D4B130(v4);
            v82[2] = (unsigned __int8 *)&v81;
            v82[3] = v83;
            v80 = v45;
            if ( sub_D51F80(v44, v3, v46, v47, v48, v49, v82, &v81, (__int64)v83)
              && (v82[6] = (unsigned __int8 *)&v81,
                  v82[7] = v83,
                  sub_D51F80(v78, v3, v50, v51, v52, v53, v82, &v81, (__int64)v83))
              && (v44 == v80
               || (v82[10] = (unsigned __int8 *)&v81,
                   v82[11] = v83,
                   sub_D51F80(v80, v3, v54, v55, v56, v57, v82, &v81, (__int64)v83))) )
            {
              v58 = sub_D47470(v4);
              v82[14] = (unsigned __int8 *)&v81;
              v82[15] = v83;
              return !sub_D51F80(v58, v3, v59, v60, v61, v62, v82, &v81, (__int64)v83);
            }
            else
            {
              return 1;
            }
          }
          return result;
        }
      }
LABEL_53:
      v42 = sub_D47470(v4);
      if ( v7 != sub_D52390(v42, v7, 0) )
        return 2;
      goto LABEL_54;
    }
    v20 = *v71;
    v21 = *(_QWORD *)(*v71 + 56);
    v22 = *v71 + 48;
    if ( v22 == v21 )
      goto LABEL_31;
    v23 = 0;
    do
    {
      v21 = *(_QWORD *)(v21 + 8);
      ++v23;
    }
    while ( v22 != v21 );
    if ( v23 == 1 )
    {
      v67 = sub_D52390(v20, v79, 0);
      v24 = sub_D52390(v20, v7, 0);
      v25 = v67;
    }
    else
    {
LABEL_31:
      v24 = *v71;
      v25 = *v71;
    }
    if ( v79 != v25 && v7 != v24 )
      break;
LABEL_66:
    v71 += 4;
  }
  if ( v73 == v69 )
    return 2;
  v26 = sub_AA4FF0(v20);
  v27 = v26;
  if ( v26 )
  {
    v27 = v26 - 24;
    v28 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v22 == v28 )
      return 2;
    goto LABEL_37;
  }
  v28 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v22 != v28 )
  {
LABEL_37:
    if ( !v28 )
      BUG();
    v29 = v28 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 >= 0xB )
      v29 = 0;
    if ( v29 != v27 )
      return 2;
  }
  v30 = sub_AA5930(v20);
  v32 = v31;
  v33 = v30;
  while ( 2 )
  {
    if ( v32 != v33 )
    {
      v34 = 32LL * *(unsigned int *)(v33 + 72);
      v35 = 8LL * (*(_DWORD *)(v33 + 4) & 0x7FFFFFF);
      v36 = v34 + v35;
      v37 = (char *)(*(_QWORD *)(v33 - 8) + v34);
      v38 = (char *)(*(_QWORD *)(v33 - 8) + v36);
      for ( i = v35 >> 5; i; --i )
      {
        if ( v17 != *(_QWORD *)v37 && v19 != *(_QWORD *)v37 )
          goto LABEL_47;
        v63 = *((_QWORD *)v37 + 1);
        if ( v17 != v63 && v19 != v63 )
        {
          v37 += 8;
          goto LABEL_47;
        }
        v64 = *((_QWORD *)v37 + 2);
        if ( v19 != v64 && v17 != v64 )
        {
          v37 += 16;
          goto LABEL_47;
        }
        v65 = *((_QWORD *)v37 + 3);
        if ( v17 != v65 && v19 != v65 )
        {
          v37 += 24;
          goto LABEL_47;
        }
        v37 += 32;
      }
      v66 = v38 - v37;
      if ( v38 - v37 == 16 )
        goto LABEL_89;
      if ( v66 != 24 )
      {
        if ( v66 != 8 )
          goto LABEL_48;
        goto LABEL_82;
      }
      if ( v19 == *(_QWORD *)v37 || v17 == *(_QWORD *)v37 )
      {
        v37 += 8;
LABEL_89:
        if ( v19 == *(_QWORD *)v37 || v17 == *(_QWORD *)v37 )
        {
          v37 += 8;
LABEL_82:
          if ( v19 == *(_QWORD *)v37 || v17 == *(_QWORD *)v37 )
            goto LABEL_48;
        }
      }
LABEL_47:
      if ( v38 != v37 )
        return 2;
LABEL_48:
      v40 = *(_QWORD *)(v33 + 32);
      if ( !v40 )
        BUG();
      v33 = 0;
      if ( *(_BYTE *)(v40 - 24) == 84 )
        v33 = v40 - 24;
      continue;
    }
    break;
  }
  if ( v7 == sub_AA56F0(v20) )
  {
    v72 = v20;
    goto LABEL_66;
  }
  return 2;
}
