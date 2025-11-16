// Function: sub_37404B0
// Address: 0x37404b0
//
__int64 __fastcall sub_37404B0(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // rsi
  int v6; // eax
  int v7; // ecx
  __int64 result; // rax
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  unsigned __int8 *v12; // rsi
  __int64 v13; // r13
  __int64 *v14; // r14
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // r10
  unsigned int v19; // r9d
  __int64 *v20; // rax
  __int64 v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char v26; // al
  int v27; // edi
  unsigned __int8 *v28; // rax
  __int64 v29; // r14
  unsigned __int64 v30; // rax
  int v31; // esi
  __int64 v32; // rdi
  unsigned __int64 v33; // rdx
  int v34; // esi
  unsigned int v35; // ecx
  __int64 *v36; // rax
  __int64 v37; // r9
  __int64 *v38; // r8
  int v39; // eax
  int v40; // eax
  int v41; // esi
  int v42; // esi
  __int64 v43; // r9
  unsigned int v44; // ecx
  __int64 v45; // rdi
  int v46; // r10d
  __int64 *v47; // r11
  int v48; // esi
  int v49; // esi
  __int64 v50; // r9
  int v51; // r10d
  unsigned int v52; // ecx
  __int64 v53; // rdi
  int v54; // eax
  int v55; // r8d
  unsigned int v56; // [rsp+8h] [rbp-58h]
  int v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+10h] [rbp-50h]
  __int64 *v60; // [rsp+18h] [rbp-48h]
  int v61; // [rsp+2Ch] [rbp-34h]

  v4 = *(_QWORD *)(a2 + 8);
  if ( !sub_3734FE0(a1) || (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), a2) )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 216) + 408LL);
    v6 = *(_DWORD *)(*(_QWORD *)(a1 + 216) + 424LL);
    if ( !v6 )
      goto LABEL_7;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 680);
    v6 = *(_DWORD *)(a1 + 696);
    if ( !v6 )
      goto LABEL_7;
  }
  v7 = v6 - 1;
  result = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = *(_QWORD *)(v5 + 16 * result);
  if ( v4 == v9 )
    return result;
  v27 = 1;
  while ( v9 != -4096 )
  {
    result = v7 & (unsigned int)(v27 + result);
    v9 = *(_QWORD *)(v5 + 16LL * (unsigned int)result);
    if ( v4 == v9 )
      return result;
    ++v27;
  }
LABEL_7:
  if ( sub_3736590((_QWORD *)a1) )
  {
    v60 = (__int64 *)a1;
    v13 = a1 + 8;
    v14 = (__int64 *)a1;
    goto LABEL_11;
  }
  v10 = *(_BYTE *)(v4 - 16);
  if ( (v10 & 2) == 0 )
  {
    v11 = v4 - 16 - 8LL * ((v10 >> 2) & 0xF);
    v12 = *(unsigned __int8 **)(v11 + 48);
    if ( v12 )
      goto LABEL_10;
LABEL_29:
    v28 = sub_373FC60((__int64 *)a1, *(unsigned __int8 **)(v11 + 8));
    v29 = *(_QWORD *)(a1 + 208);
    v13 = (__int64)v28;
    v30 = sub_32150B0((unsigned __int64)v28);
    v31 = *(_DWORD *)(v29 + 696);
    v32 = *(_QWORD *)(v29 + 680);
    v33 = v30;
    if ( v31 )
    {
      v34 = v31 - 1;
      v35 = v34 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v36 = (__int64 *)(v32 + 16LL * v35);
      v37 = *v36;
      if ( v33 == *v36 )
      {
LABEL_31:
        v14 = (__int64 *)v36[1];
        v60 = v14;
        goto LABEL_11;
      }
      v54 = 1;
      while ( v37 != -4096 )
      {
        v55 = v54 + 1;
        v35 = v34 & (v54 + v35);
        v36 = (__int64 *)(v32 + 16LL * v35);
        v37 = *v36;
        if ( v33 == *v36 )
          goto LABEL_31;
        v54 = v55;
      }
    }
    v60 = 0;
    v14 = 0;
    goto LABEL_11;
  }
  v11 = *(_QWORD *)(v4 - 32);
  v12 = *(unsigned __int8 **)(v11 + 48);
  if ( !v12 )
    goto LABEL_29;
LABEL_10:
  v60 = (__int64 *)a1;
  v13 = a1 + 8;
  sub_3250680((_QWORD *)a1, v12, 0);
  v14 = (__int64 *)a1;
LABEL_11:
  v15 = sub_324C6D0(v60, 46, v13, 0);
  if ( !sub_3734FE0((__int64)v14) || (v26 = sub_321F6A0(v14[26], 46), v16 = (__int64)(v14 + 84), v26) )
    v16 = v14[27] + 400;
  v17 = *(_DWORD *)(v16 + 24);
  if ( !v17 )
  {
    ++*(_QWORD *)v16;
    goto LABEL_42;
  }
  v18 = *(_QWORD *)(v16 + 8);
  v19 = (v17 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v20 = (__int64 *)(v18 + 16LL * v19);
  v21 = *v20;
  if ( v4 != *v20 )
  {
    v57 = 1;
    v38 = 0;
    while ( v21 != -4096 )
    {
      if ( v21 == -8192 && !v38 )
        v38 = v20;
      v19 = (v17 - 1) & (v57 + v19);
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v4 == *v20 )
        goto LABEL_15;
      ++v57;
    }
    if ( !v38 )
      v38 = v20;
    v39 = *(_DWORD *)(v16 + 16);
    ++*(_QWORD *)v16;
    v40 = v39 + 1;
    if ( 4 * v40 < 3 * v17 )
    {
      if ( v17 - *(_DWORD *)(v16 + 20) - v40 > v17 >> 3 )
      {
LABEL_38:
        *(_DWORD *)(v16 + 16) = v40;
        if ( *v38 != -4096 )
          --*(_DWORD *)(v16 + 20);
        *v38 = v4;
        v22 = v38 + 1;
        v38[1] = 0;
        goto LABEL_16;
      }
      v59 = v16;
      v56 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
      sub_373B830(v16, v17);
      v16 = v59;
      v48 = *(_DWORD *)(v59 + 24);
      if ( v48 )
      {
        v49 = v48 - 1;
        v50 = *(_QWORD *)(v59 + 8);
        v51 = 1;
        v47 = 0;
        v52 = v49 & v56;
        v38 = (__int64 *)(v50 + 16LL * (v49 & v56));
        v53 = *v38;
        v40 = *(_DWORD *)(v59 + 16) + 1;
        if ( v4 == *v38 )
          goto LABEL_38;
        while ( v53 != -4096 )
        {
          if ( !v47 && v53 == -8192 )
            v47 = v38;
          v52 = v49 & (v51 + v52);
          v38 = (__int64 *)(v50 + 16LL * v52);
          v53 = *v38;
          if ( v4 == *v38 )
            goto LABEL_38;
          ++v51;
        }
        goto LABEL_46;
      }
      goto LABEL_72;
    }
LABEL_42:
    v58 = v16;
    sub_373B830(v16, 2 * v17);
    v16 = v58;
    v41 = *(_DWORD *)(v58 + 24);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(v58 + 8);
      v44 = v42 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v38 = (__int64 *)(v43 + 16LL * v44);
      v45 = *v38;
      v40 = *(_DWORD *)(v58 + 16) + 1;
      if ( v4 == *v38 )
        goto LABEL_38;
      v46 = 1;
      v47 = 0;
      while ( v45 != -4096 )
      {
        if ( !v47 && v45 == -8192 )
          v47 = v38;
        v44 = v42 & (v46 + v44);
        v38 = (__int64 *)(v43 + 16LL * v44);
        v45 = *v38;
        if ( v4 == *v38 )
          goto LABEL_38;
        ++v46;
      }
LABEL_46:
      if ( v47 )
        v38 = v47;
      goto LABEL_38;
    }
LABEL_72:
    ++*(_DWORD *)(v16 + 16);
    BUG();
  }
LABEL_15:
  v22 = v20 + 1;
LABEL_16:
  *v22 = v15;
  sub_37373C0(v14, v4, v15);
  if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 4u )
    v61 = 65569;
  else
    HIWORD(v61) = 0;
  sub_32498F0(v60, (unsigned __int64 **)(v15 + 8), 32, v61, 1);
  result = sub_373DE40((__int64)v14, a2, v15, v23, v24, v25);
  if ( result )
    return sub_32494F0(v60, v15, 100, result);
  return result;
}
