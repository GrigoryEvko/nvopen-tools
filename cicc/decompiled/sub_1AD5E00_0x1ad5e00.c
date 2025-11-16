// Function: sub_1AD5E00
// Address: 0x1ad5e00
//
__int64 __fastcall sub_1AD5E00(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rdi
  unsigned int v5; // eax
  bool v6; // zf
  __int64 *v7; // rsi
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // r13
  int v11; // eax
  __int64 v12; // rdx
  __int64 *v13; // rax
  char v14; // r15
  unsigned int v15; // esi
  __int64 v16; // r8
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // r14
  _QWORD *v23; // r8
  int v24; // r9d
  _QWORD *v25; // r13
  unsigned __int8 v26; // al
  int v28; // r11d
  __int64 *v29; // r10
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // rax
  __int64 v33; // rcx
  unsigned int v34; // esi
  _QWORD *v35; // rdx
  int v36; // eax
  unsigned int v37; // eax
  __int64 *v38; // rdx
  __int64 v39; // rdx
  int v40; // edi
  int v41; // edi
  __int64 v42; // r9
  unsigned int v43; // esi
  __int64 v44; // r8
  int v45; // r11d
  __int64 *v46; // r10
  __int64 v47; // rdi
  int v48; // esi
  int v49; // esi
  __int64 v50; // r8
  __int64 *v51; // r9
  unsigned int v52; // r14d
  int v53; // r10d
  __int64 v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // r15
  __int64 *v57; // r14
  __int64 *v58; // r15
  __int64 v59; // rax
  __int64 v60; // rbx
  _QWORD *v61; // rax
  int v62; // r8d
  _QWORD *v63; // r9
  unsigned __int8 v64; // dl
  __int64 v65; // rdx
  __int64 v66; // rdi
  unsigned int v67; // esi
  _QWORD *v68; // rcx
  int v69; // ecx
  __int64 v70; // rdx
  __int64 *v71; // rax
  int v72; // edx
  __int64 v73; // rax
  int v74; // r10d
  _QWORD *v75; // [rsp+0h] [rbp-A0h]
  __int64 v76; // [rsp+10h] [rbp-90h]
  __int64 v77; // [rsp+10h] [rbp-90h]
  __int64 v78; // [rsp+10h] [rbp-90h]
  _QWORD *v80; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v81; // [rsp+28h] [rbp-78h]
  unsigned int v82; // [rsp+2Ch] [rbp-74h]
  _QWORD v83[14]; // [rsp+30h] [rbp-70h] BYREF

  v3 = a1;
  v83[0] = a1;
  v4 = v83;
  v80 = v83;
  v5 = 0;
  v6 = *(_BYTE *)(v3 + 16) == 34;
  v82 = 8;
  v81 = 0;
  if ( !v6 )
    goto LABEL_24;
LABEL_2:
  v7 = (__int64 *)v3;
  v8 = *(_BYTE *)(v3 + 23) & 0x40;
  if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
  {
    if ( v8 )
      v9 = *(_QWORD *)(v3 - 8);
    else
      v9 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    v10 = sub_157ED20(*(_QWORD *)(v9 + 24));
    goto LABEL_6;
  }
  v55 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
  if ( v8 )
  {
    v56 = *(_QWORD *)(v3 - 8);
    v57 = (__int64 *)(v56 + 24);
    v7 = (__int64 *)(v56 + v55);
  }
  else
  {
    v57 = (__int64 *)(v3 - v55 + 24);
  }
  if ( v7 != v57 )
  {
    v78 = v3;
    v58 = v7;
    while ( 1 )
    {
      v59 = sub_15A5110(*v57);
      v60 = *(_QWORD *)(sub_157ED20(v59) + 8);
      if ( v60 )
        break;
LABEL_101:
      v57 += 3;
      if ( v58 == v57 )
        goto LABEL_31;
    }
    while ( 1 )
    {
      v61 = sub_1648700(v60);
      v64 = *((_BYTE *)v61 + 16);
      if ( v64 <= 0x17u || v64 != 34 && v64 != 73 )
        goto LABEL_87;
      v65 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v65 )
        goto LABEL_98;
      v62 = v65 - 1;
      v66 = *(_QWORD *)(a2 + 8);
      v67 = (v65 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v68 = (_QWORD *)(v66 + 16LL * v67);
      v63 = (_QWORD *)*v68;
      if ( v61 != (_QWORD *)*v68 )
      {
        v69 = 1;
        while ( v63 != (_QWORD *)-8LL )
        {
          v74 = v69 + 1;
          v67 = v62 & (v69 + v67);
          v68 = (_QWORD *)(v66 + 16LL * v67);
          v63 = (_QWORD *)*v68;
          if ( v61 == (_QWORD *)*v68 )
            goto LABEL_92;
          v69 = v74;
        }
        goto LABEL_98;
      }
LABEL_92:
      if ( v68 == (_QWORD *)(v66 + 16 * v65) )
      {
LABEL_98:
        v70 = v81;
        if ( v81 >= v82 )
        {
          v75 = v61;
          sub_16CD150((__int64)&v80, v83, 0, 8, v62, (int)v63);
          v70 = v81;
          v61 = v75;
        }
        v80[v70] = v61;
        ++v81;
        v60 = *(_QWORD *)(v60 + 8);
        if ( !v60 )
          goto LABEL_101;
      }
      else
      {
        v10 = v68[1];
        if ( v10 && *(_BYTE *)(v10 + 16) == 16 )
        {
          v3 = v78;
LABEL_6:
          if ( v10 )
          {
            v11 = *(unsigned __int8 *)(v10 + 16);
            v12 = 0;
            if ( (unsigned __int8)v11 <= 0x17u )
              goto LABEL_12;
            if ( (unsigned int)(v11 - 73) <= 1 )
              goto LABEL_61;
            goto LABEL_9;
          }
LABEL_31:
          v5 = v81;
          v4 = v80;
          goto LABEL_32;
        }
LABEL_87:
        v60 = *(_QWORD *)(v60 + 8);
        if ( !v60 )
          goto LABEL_101;
      }
    }
  }
  do
  {
LABEL_32:
    if ( !v5 )
      goto LABEL_33;
LABEL_23:
    v21 = v5--;
    v3 = v4[v21 - 1];
    v81 = v5;
    if ( *(_BYTE *)(v3 + 16) == 34 )
      goto LABEL_2;
LABEL_24:
    v22 = *(_QWORD *)(v3 + 8);
  }
  while ( !v22 );
  while ( 1 )
  {
    v25 = sub_1648700(v22);
    v26 = *((_BYTE *)v25 + 16);
    if ( v26 <= 0x17u )
      goto LABEL_30;
    if ( v26 == 32 )
    {
      if ( (*((_BYTE *)v25 + 18) & 1) != 0 && (v47 = v25[3 * (1LL - (*((_DWORD *)v25 + 5) & 0xFFFFFFF))]) != 0 )
      {
        v10 = sub_157ED20(v47);
      }
      else
      {
        v71 = (__int64 *)sub_16498A0(v3);
        v10 = sub_1594470(v71);
      }
      goto LABEL_6;
    }
    if ( v26 == 29 )
    {
      v10 = sub_157ED20(*(v25 - 3));
      goto LABEL_54;
    }
    if ( v26 != 34 && v26 != 73 )
      goto LABEL_30;
    v32 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v32 )
      goto LABEL_109;
    v33 = *(_QWORD *)(a2 + 8);
    v34 = (v32 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v35 = (_QWORD *)(v33 + 16LL * v34);
    v23 = (_QWORD *)*v35;
    if ( v25 == (_QWORD *)*v35 )
      break;
    v72 = 1;
    while ( v23 != (_QWORD *)-8LL )
    {
      v24 = v72 + 1;
      v34 = (v32 - 1) & (v72 + v34);
      v35 = (_QWORD *)(v33 + 16LL * v34);
      v23 = (_QWORD *)*v35;
      if ( v25 == (_QWORD *)*v35 )
        goto LABEL_52;
      v72 = v24;
    }
LABEL_109:
    v73 = v81;
    if ( v81 >= v82 )
    {
      sub_16CD150((__int64)&v80, v83, 0, 8, (int)v23, v24);
      v73 = v81;
    }
    v80[v73] = v25;
    ++v81;
LABEL_30:
    v22 = *(_QWORD *)(v22 + 8);
    if ( !v22 )
      goto LABEL_31;
  }
LABEL_52:
  if ( v35 == (_QWORD *)(v33 + 16 * v32) )
    goto LABEL_109;
  v10 = v35[1];
  if ( !v10 )
    goto LABEL_30;
LABEL_54:
  v36 = *(unsigned __int8 *)(v10 + 16);
  if ( (unsigned __int8)v36 <= 0x17u )
  {
    v12 = 0;
    goto LABEL_12;
  }
  v37 = v36 - 73;
  if ( v37 <= 1 )
  {
    v39 = *(_QWORD *)(v10 - 24);
  }
  else
  {
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      v38 = *(__int64 **)(v10 - 8);
    else
      v38 = (__int64 *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    v39 = *v38;
    if ( !v39 )
      goto LABEL_9;
  }
  if ( v39 == v3 )
    goto LABEL_30;
  if ( v37 <= 1 )
  {
LABEL_61:
    v12 = *(_QWORD *)(v10 - 24);
    goto LABEL_12;
  }
LABEL_9:
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v13 = *(__int64 **)(v10 - 8);
  else
    v13 = (__int64 *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v12 = *v13;
LABEL_12:
  v14 = 0;
  while ( 2 )
  {
    if ( v12 != v3 )
    {
      if ( *(_BYTE *)(v3 + 16) == 74 )
        goto LABEL_40;
      v15 = *(_DWORD *)(a2 + 24);
      if ( v15 )
      {
        v16 = *(_QWORD *)(a2 + 8);
        v17 = (v15 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v3 == *v18 )
          goto LABEL_17;
        v28 = 1;
        v29 = 0;
        while ( v19 != -8 )
        {
          if ( !v29 && v19 == -16 )
            v29 = v18;
          v17 = (v15 - 1) & (v28 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v3 == *v18 )
            goto LABEL_17;
          ++v28;
        }
        v30 = *(_DWORD *)(a2 + 16);
        if ( v29 )
          v18 = v29;
        ++*(_QWORD *)a2;
        v31 = v30 + 1;
        if ( 4 * v31 < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(a2 + 20) - v31 <= v15 >> 3 )
          {
            v77 = v12;
            sub_19566A0(a2, v15);
            v48 = *(_DWORD *)(a2 + 24);
            if ( !v48 )
            {
LABEL_134:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v49 = v48 - 1;
            v50 = *(_QWORD *)(a2 + 8);
            v51 = 0;
            v52 = v49 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
            v12 = v77;
            v53 = 1;
            v31 = *(_DWORD *)(a2 + 16) + 1;
            v18 = (__int64 *)(v50 + 16LL * v52);
            v54 = *v18;
            if ( v3 != *v18 )
            {
              while ( v54 != -8 )
              {
                if ( v54 == -16 && !v51 )
                  v51 = v18;
                v52 = v49 & (v53 + v52);
                v18 = (__int64 *)(v50 + 16LL * v52);
                v54 = *v18;
                if ( v3 == *v18 )
                  goto LABEL_47;
                ++v53;
              }
              if ( v51 )
                v18 = v51;
            }
          }
          goto LABEL_47;
        }
      }
      else
      {
        ++*(_QWORD *)a2;
      }
      v76 = v12;
      sub_19566A0(a2, 2 * v15);
      v40 = *(_DWORD *)(a2 + 24);
      if ( !v40 )
        goto LABEL_134;
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a2 + 8);
      v12 = v76;
      v43 = v41 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v31 = *(_DWORD *)(a2 + 16) + 1;
      v18 = (__int64 *)(v42 + 16LL * v43);
      v44 = *v18;
      if ( v3 != *v18 )
      {
        v45 = 1;
        v46 = 0;
        while ( v44 != -8 )
        {
          if ( v44 == -16 && !v46 )
            v46 = v18;
          v43 = v41 & (v45 + v43);
          v18 = (__int64 *)(v42 + 16LL * v43);
          v44 = *v18;
          if ( v3 == *v18 )
            goto LABEL_47;
          ++v45;
        }
        if ( v46 )
          v18 = v46;
      }
LABEL_47:
      *(_DWORD *)(a2 + 16) = v31;
      if ( *v18 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v18 = v3;
      v18[1] = 0;
LABEL_17:
      v18[1] = v10;
      v14 |= v3 == a1;
      v20 = *(unsigned __int8 *)(v3 + 16);
      if ( (unsigned __int8)v20 > 0x17u && (unsigned int)(v20 - 73) <= 1 )
      {
LABEL_40:
        v3 = *(_QWORD *)(v3 - 24);
      }
      else if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
      {
        v3 = **(_QWORD **)(v3 - 8);
        if ( !v3 )
          goto LABEL_38;
      }
      else
      {
        v3 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
        if ( !v3 )
LABEL_38:
          BUG();
      }
      if ( *(_BYTE *)(v3 + 16) <= 0x17u )
        break;
      continue;
    }
    break;
  }
  v4 = v80;
  if ( !v14 )
  {
    v5 = v81;
    if ( v81 )
      goto LABEL_23;
LABEL_33:
    v10 = 0;
  }
  if ( v4 != v83 )
    _libc_free((unsigned __int64)v4);
  return v10;
}
