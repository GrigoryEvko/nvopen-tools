// Function: sub_1BF60E0
// Address: 0x1bf60e0
//
char __fastcall sub_1BF60E0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  _QWORD **v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r8
  char v20; // al
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // r15
  unsigned int v24; // eax
  __int64 v25; // r8
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // r15d
  __int64 v31; // rdi
  _QWORD *v32; // rsi
  _QWORD *v33; // rax
  _QWORD *v34; // rbx
  __int64 v35; // r8
  char v36; // di
  unsigned int v37; // esi
  __int64 v38; // rdx
  __int64 v39; // rax
  _QWORD *v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 *v43; // rax
  __int64 *v44; // rdi
  unsigned int v45; // r8d
  __int64 *v46; // rcx
  _QWORD *v47; // rcx
  unsigned int v48; // edi
  _QWORD *v49; // rdx
  unsigned int v50; // r8d
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 *v53; // rdi
  unsigned int v54; // r8d
  __int64 *v55; // rcx
  __int64 v56; // rax
  __int64 v58; // [rsp+8h] [rbp-48h]
  __int64 v59; // [rsp+8h] [rbp-48h]
  __int64 v60; // [rsp+10h] [rbp-40h]
  __int64 v61[7]; // [rsp+18h] [rbp-38h] BYREF

  v61[0] = (__int64)a2;
  v9 = sub_1BF5C70(a1 + 104, v61, a3, a4, a5, a6);
  v13 = *(_QWORD *)(v9 + 16);
  v14 = v9;
  v15 = *(_QWORD *)(a3 + 16);
  if ( v13 != v15 )
  {
    if ( v13 != 0 && v13 != -8 && v13 != -16 )
    {
      sub_1649B30((_QWORD *)v14);
      v15 = *(_QWORD *)(a3 + 16);
    }
    *(_QWORD *)(v14 + 16) = v15;
    LOBYTE(v10) = v15 != -8;
    LOBYTE(v13) = v15 != 0;
    if ( ((v15 != 0) & (unsigned __int8)v10) != 0 && v15 != -16 )
      sub_1649AC0((unsigned __int64 *)v14, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v16 = a3 + 48;
  *(_DWORD *)(v14 + 24) = *(_DWORD *)(a3 + 24);
  *(_QWORD *)(v14 + 32) = *(_QWORD *)(a3 + 32);
  *(_QWORD *)(v14 + 40) = *(_QWORD *)(a3 + 40);
  sub_1BF0AA0(v14 + 48, a3 + 48, v13, v10, v11, v12);
  if ( !*(_DWORD *)(a3 + 56) )
    goto LABEL_9;
  v16 = **(_QWORD **)(a3 + 48);
  v43 = *(__int64 **)(a1 + 168);
  if ( *(__int64 **)(a1 + 176) != v43 )
  {
LABEL_43:
    sub_16CCBA0(a1 + 160, v16);
    goto LABEL_9;
  }
  v44 = &v43[*(unsigned int *)(a1 + 188)];
  v45 = *(_DWORD *)(a1 + 188);
  if ( v43 == v44 )
  {
LABEL_81:
    if ( v45 < *(_DWORD *)(a1 + 184) )
    {
      *(_DWORD *)(a1 + 188) = v45 + 1;
      *v44 = v16;
      ++*(_QWORD *)(a1 + 160);
      goto LABEL_9;
    }
    goto LABEL_43;
  }
  v46 = 0;
  while ( v16 != *v43 )
  {
    if ( *v43 == -2 )
      v46 = v43;
    if ( v44 == ++v43 )
    {
      if ( !v46 )
        goto LABEL_81;
      *v46 = v16;
      --*(_DWORD *)(a1 + 192);
      ++*(_QWORD *)(a1 + 160);
      break;
    }
  }
LABEL_9:
  v17 = *(_QWORD ***)v61[0];
  v18 = sub_15F2050(v61[0]);
  v19 = sub_1632FA0(v18);
  v20 = *((_BYTE *)v17 + 8);
  if ( (unsigned __int8)(v20 - 1) > 5u )
  {
    v23 = *(_QWORD *)(a1 + 368);
    if ( v23 )
    {
      if ( v20 == 15 )
      {
        v16 = (__int64)v17;
        v59 = v19;
        v56 = sub_15A9650(v19, (__int64)v17);
        v25 = v59;
        v60 = v56;
      }
      else
      {
        v58 = v19;
        v24 = sub_16431D0((__int64)v17);
        v60 = (__int64)v17;
        v25 = v58;
        if ( v24 <= 0x1F )
        {
          v52 = sub_1643350(*v17);
          v25 = v58;
          v60 = v52;
        }
      }
      if ( *(_BYTE *)(v23 + 8) == 15 )
      {
        v16 = v23;
        v23 = sub_15A9650(v25, v23);
        v26 = sub_16431D0(v23);
      }
      else
      {
        v26 = sub_16431D0(v23);
        if ( v26 <= 0x1F )
        {
          v23 = sub_1643350(*(_QWORD **)v23);
          v26 = sub_16431D0(v23);
        }
      }
      if ( v26 < (unsigned int)sub_16431D0(v60) )
        v23 = v60;
      *(_QWORD *)(a1 + 368) = v23;
      if ( *(_DWORD *)(a3 + 24) != 1 )
        goto LABEL_11;
      goto LABEL_23;
    }
    if ( v20 == 15 )
    {
      v16 = (__int64)v17;
      v51 = sub_15A9650(v19, (__int64)v17);
    }
    else
    {
      v50 = sub_16431D0((__int64)v17);
      v51 = (__int64)v17;
      if ( v50 <= 0x1F )
        v51 = sub_1643350(*v17);
    }
    *(_QWORD *)(a1 + 368) = v51;
  }
  if ( *(_DWORD *)(a3 + 24) != 1 )
    goto LABEL_11;
LABEL_23:
  if ( !sub_1B16970(a3) )
    goto LABEL_11;
  v27 = sub_1B16970(a3);
  v30 = *(_DWORD *)(v27 + 32);
  if ( v30 <= 0x40 )
  {
    if ( *(_QWORD *)(v27 + 24) != 1 )
      goto LABEL_11;
  }
  else if ( (unsigned int)sub_16A57B0(v27 + 24) != v30 - 1 )
  {
    goto LABEL_11;
  }
  v31 = *(_QWORD *)(a3 + 16);
  if ( *(_BYTE *)(v31 + 16) <= 0x10u
    && sub_1593BB0(v31, v16, v28, v29)
    && (!*(_QWORD *)(a1 + 64) || *(_QWORD ***)(a1 + 368) == v17) )
  {
    *(_QWORD *)(a1 + 64) = v61[0];
  }
LABEL_11:
  v21 = sub_1458800(*(_QWORD *)(a1 + 16));
  LOBYTE(v22) = sub_1452CB0(v21);
  if ( !(_BYTE)v22 )
    return (char)v22;
  v32 = (_QWORD *)v61[0];
  v33 = *(_QWORD **)(a4 + 8);
  if ( *(_QWORD **)(a4 + 16) != v33 )
    goto LABEL_32;
  v47 = &v33[*(unsigned int *)(a4 + 28)];
  v48 = *(_DWORD *)(a4 + 28);
  if ( v33 == v47 )
  {
LABEL_79:
    if ( v48 >= *(_DWORD *)(a4 + 24) )
    {
LABEL_32:
      sub_16CCBA0(a4, v61[0]);
      v34 = (_QWORD *)v61[0];
      goto LABEL_33;
    }
    *(_DWORD *)(a4 + 28) = v48 + 1;
    *v47 = v32;
    v34 = (_QWORD *)v61[0];
    ++*(_QWORD *)a4;
  }
  else
  {
    v49 = 0;
    while ( 1 )
    {
      v34 = (_QWORD *)*v33;
      if ( v61[0] == *v33 )
        break;
      if ( v34 == (_QWORD *)-2LL )
        v49 = v33;
      if ( v47 == ++v33 )
      {
        if ( !v49 )
          goto LABEL_79;
        *v49 = v61[0];
        v34 = v32;
        --*(_DWORD *)(a4 + 32);
        ++*(_QWORD *)a4;
        break;
      }
    }
  }
LABEL_33:
  v35 = sub_13FCB50(*(_QWORD *)a1);
  v36 = *((_BYTE *)v34 + 23) & 0x40;
  v37 = *((_DWORD *)v34 + 5) & 0xFFFFFFF;
  if ( !v37 )
  {
LABEL_64:
    v41 = 0x2FFFFFFFDLL;
    if ( v36 )
      goto LABEL_40;
LABEL_65:
    v42 = v34[v41 + -3 * v37];
    v22 = *(__int64 **)(a4 + 8);
    if ( *(__int64 **)(a4 + 16) != v22 )
      goto LABEL_41;
    goto LABEL_66;
  }
  v38 = 24LL * *((unsigned int *)v34 + 14) + 8;
  v39 = 0;
  while ( 1 )
  {
    v40 = &v34[-3 * v37];
    if ( v36 )
      v40 = (_QWORD *)*(v34 - 1);
    if ( v35 == *(_QWORD *)((char *)v40 + v38) )
      break;
    ++v39;
    v38 += 8;
    if ( v37 == (_DWORD)v39 )
      goto LABEL_64;
  }
  v41 = 3 * v39;
  if ( !v36 )
    goto LABEL_65;
LABEL_40:
  v42 = *(_QWORD *)(*(v34 - 1) + v41 * 8);
  v22 = *(__int64 **)(a4 + 8);
  if ( *(__int64 **)(a4 + 16) != v22 )
  {
LABEL_41:
    LOBYTE(v22) = (unsigned __int8)sub_16CCBA0(a4, v42);
    return (char)v22;
  }
LABEL_66:
  v53 = &v22[*(unsigned int *)(a4 + 28)];
  v54 = *(_DWORD *)(a4 + 28);
  if ( v22 == v53 )
  {
LABEL_83:
    if ( v54 < *(_DWORD *)(a4 + 24) )
    {
      *(_DWORD *)(a4 + 28) = v54 + 1;
      *v53 = v42;
      ++*(_QWORD *)a4;
      return (char)v22;
    }
    goto LABEL_41;
  }
  v55 = 0;
  while ( v42 != *v22 )
  {
    if ( *v22 == -2 )
      v55 = v22;
    if ( v53 == ++v22 )
    {
      if ( !v55 )
        goto LABEL_83;
      *v55 = v42;
      --*(_DWORD *)(a4 + 32);
      ++*(_QWORD *)a4;
      return (char)v22;
    }
  }
  return (char)v22;
}
