// Function: sub_2B68AE0
// Address: 0x2b68ae0
//
__int64 __fastcall sub_2B68AE0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 *v5; // r13
  unsigned __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // rsi
  int v16; // eax
  int v17; // ecx
  unsigned int v18; // eax
  __int64 *v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 *v32; // rcx
  __int64 v33; // r14
  unsigned __int8 *v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rcx
  __int64 *v39; // r12
  int v40; // ecx
  unsigned __int8 *v41; // rcx
  __int64 v42; // rsi
  __int64 *v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rsi
  _BYTE *v46; // rdx
  unsigned __int64 v47; // rax
  int v48; // esi
  int v49; // edi
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 *v52; // [rsp+20h] [rbp-90h]
  __int128 v54; // [rsp+30h] [rbp-80h] BYREF
  __int128 v55; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v56; // [rsp+50h] [rbp-60h] BYREF
  __int64 v57; // [rsp+58h] [rbp-58h]
  _BYTE v58[80]; // [rsp+60h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 240) + 80LL * a3;
  v5 = *(__int64 **)v4;
  v6 = *(unsigned int *)(v4 + 8);
  v7 = sub_2B5F980(*(__int64 **)v4, v6, *(__int64 **)(a1 + 3304));
  v50 = v7;
  if ( !v7 )
  {
    v41 = (unsigned __int8 *)*v5;
    if ( *(_BYTE *)(*(_QWORD *)(*v5 + 8) + 8LL) != 14 )
      goto LABEL_42;
    v42 = (__int64)&v5[v6];
    v43 = (__int64 *)sub_2B0C150((_BYTE **)v5, v42);
    if ( v43 == (__int64 *)v42 )
      goto LABEL_42;
    goto LABEL_55;
  }
  if ( !v8 )
  {
    v35 = (unsigned __int8 *)*v5;
    if ( *(_BYTE *)(*(_QWORD *)(*v5 + 8) + 8LL) != 14
      || (v45 = (__int64)&v5[v6], v43 = (__int64 *)sub_2B0C150((_BYTE **)v5, v45), (__int64 *)v45 == v43) )
    {
LABEL_27:
      v50 = (__int64)v35;
      if ( !(unsigned __int8)sub_2B0D8B0(v35) )
        goto LABEL_7;
LABEL_13:
      v26 = (unsigned int)(*(_DWORD *)(a2 + 200) + 1);
      v27 = (__int64 *)(*(_QWORD *)a1 + 8 * v26);
      v28 = 8 * (*(unsigned int *)(a1 + 8) - v26);
      v29 = &v27[(unsigned __int64)v28 / 8];
      v30 = v28 >> 5;
      v31 = v28 >> 3;
      if ( v30 > 0 )
      {
        v32 = &v27[4 * v30];
        while ( 1 )
        {
          v33 = *v27;
          if ( ((*(_DWORD *)(*v27 + 104) - 3) & 0xFFFFFFFD) == 0
            && a3 == *(_DWORD *)(v33 + 192)
            && a2 == *(_QWORD *)(v33 + 184) )
          {
            return v33;
          }
          v33 = v27[1];
          if ( ((*(_DWORD *)(v33 + 104) - 3) & 0xFFFFFFFD) == 0
            && a3 == *(_DWORD *)(v33 + 192)
            && a2 == *(_QWORD *)(v33 + 184) )
          {
            return v33;
          }
          v33 = v27[2];
          if ( ((*(_DWORD *)(v33 + 104) - 3) & 0xFFFFFFFD) == 0
            && a3 == *(_DWORD *)(v33 + 192)
            && a2 == *(_QWORD *)(v33 + 184) )
          {
            return v33;
          }
          v33 = v27[3];
          if ( ((*(_DWORD *)(v33 + 104) - 3) & 0xFFFFFFFD) == 0
            && a3 == *(_DWORD *)(v33 + 192)
            && a2 == *(_QWORD *)(v33 + 184) )
          {
            return v33;
          }
          v27 += 4;
          if ( v32 == v27 )
          {
            v31 = v29 - v27;
            break;
          }
        }
      }
      if ( v31 != 2 )
      {
        if ( v31 != 3 )
        {
          if ( v31 != 1 )
            return *v29;
          goto LABEL_87;
        }
        v33 = *v27;
        if ( ((*(_DWORD *)(*v27 + 104) - 3) & 0xFFFFFFFD) == 0
          && a3 == *(_DWORD *)(v33 + 192)
          && a2 == *(_QWORD *)(v33 + 184) )
        {
          return v33;
        }
        ++v27;
      }
      v33 = *v27;
      if ( ((*(_DWORD *)(*v27 + 104) - 3) & 0xFFFFFFFD) == 0
        && a3 == *(_DWORD *)(v33 + 192)
        && a2 == *(_QWORD *)(v33 + 184) )
      {
        return v33;
      }
      ++v27;
LABEL_87:
      v33 = *v27;
      if ( ((*(_DWORD *)(*v27 + 104) - 3) & 0xFFFFFFFD) == 0
        && a3 == *(_DWORD *)(v33 + 192)
        && a2 == *(_QWORD *)(v33 + 184) )
      {
        return v33;
      }
      return *v29;
    }
LABEL_55:
    v7 = sub_2B5F980(v43, 1u, *(__int64 **)(a1 + 3304));
    v50 = v7;
    if ( v7 )
    {
      if ( v44 )
        goto LABEL_3;
      v35 = (unsigned __int8 *)*v5;
      goto LABEL_27;
    }
    v41 = (unsigned __int8 *)*v5;
LABEL_42:
    v50 = (__int64)v41;
    if ( !(unsigned __int8)sub_2B0D8B0(v41) )
      goto LABEL_7;
    goto LABEL_13;
  }
LABEL_3:
  if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
  {
    v10 = a1 + 96;
    v11 = 3;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 96);
    v40 = *(_DWORD *)(a1 + 104);
    if ( !v40 )
      goto LABEL_7;
    v11 = (unsigned int)(v40 - 1);
  }
  v12 = (unsigned int)v11 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v13 = v10 + 72 * v12;
  v14 = *(_QWORD *)v13;
  if ( *(_QWORD *)v13 == v7 )
  {
LABEL_6:
    *(_QWORD *)&v54 = &v55;
    *((_QWORD *)&v54 + 1) = 0x600000000LL;
    if ( *(_DWORD *)(v13 + 16) )
    {
      sub_2B0C870((__int64)&v54, v13 + 8, v12, v11, v14, v9);
      v38 = v54;
      v52 = (__int64 *)(v54 + 8LL * DWORD2(v54));
      if ( (__int64 *)v54 == v52 )
      {
LABEL_51:
        if ( v52 != (__int64 *)&v55 )
          _libc_free((unsigned __int64)v52);
      }
      else
      {
        v39 = (__int64 *)v54;
        while ( 1 )
        {
          v33 = *v39;
          if ( sub_2B31C30(*v39, (char *)v5, v6, v38, v36, v37) )
            break;
          if ( v52 == ++v39 )
          {
            v52 = (__int64 *)v54;
            goto LABEL_51;
          }
        }
        if ( (__int128 *)v54 != &v55 )
          _libc_free(v54);
        if ( v33 && *(_QWORD *)(v33 + 184) == a2 && *(_DWORD *)(v33 + 192) == a3 )
          return v33;
      }
    }
  }
  else
  {
    v48 = 1;
    while ( v14 != -4096 )
    {
      v9 = (unsigned int)(v48 + 1);
      v12 = (unsigned int)v11 & (v48 + (_DWORD)v12);
      v13 = v10 + 72LL * (unsigned int)v12;
      v14 = *(_QWORD *)v13;
      if ( *(_QWORD *)v13 == v7 )
        goto LABEL_6;
      v48 = v9;
    }
  }
LABEL_7:
  v15 = *(_QWORD *)(a1 + 1176);
  v16 = *(_DWORD *)(a1 + 1192);
  if ( !v16 )
  {
LABEL_69:
    v54 = 0;
    v55 = 0;
LABEL_12:
    sub_C7D6A0(*((__int64 *)&v54 + 1), 8LL * DWORD2(v55), 8);
    goto LABEL_13;
  }
  v17 = v16 - 1;
  v18 = (v16 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
  v19 = (__int64 *)(v15 + 88LL * v18);
  v20 = *v19;
  if ( v50 != *v19 )
  {
    v49 = 1;
    while ( v20 != -4096 )
    {
      v18 = v17 & (v49 + v18);
      v19 = (__int64 *)(v15 + 88LL * v18);
      v20 = *v19;
      if ( v50 == *v19 )
        goto LABEL_9;
      ++v49;
    }
    goto LABEL_69;
  }
LABEL_9:
  v54 = 0u;
  v55 = 0u;
  sub_C7D6A0(0, 0, 8);
  v24 = *((unsigned int *)v19 + 8);
  DWORD2(v55) = v24;
  if ( (_DWORD)v24 )
  {
    *((_QWORD *)&v54 + 1) = sub_C7D670(8 * v24, 8);
    *(_QWORD *)&v55 = v19[3];
    memcpy(*((void **)&v54 + 1), (const void *)v19[2], 8LL * DWORD2(v55));
  }
  else
  {
    *((_QWORD *)&v54 + 1) = 0;
    *(_QWORD *)&v55 = 0;
  }
  v56 = v58;
  v57 = 0x400000000LL;
  v25 = *((unsigned int *)v19 + 12);
  if ( !(_DWORD)v25 )
    goto LABEL_12;
  sub_2B0C210((__int64)&v56, (__int64)(v19 + 5), v25, v21, v22, v23);
  v46 = &v56[8 * (unsigned int)v57];
  if ( v46 == v56 )
  {
LABEL_67:
    if ( v56 != v58 )
      _libc_free((unsigned __int64)v56);
    goto LABEL_12;
  }
  v47 = (unsigned __int64)v56;
  while ( 1 )
  {
    v33 = *(_QWORD *)v47;
    if ( *(_DWORD *)(*(_QWORD *)v47 + 192LL) == a3 && *(_QWORD *)(v33 + 184) == a2 )
      break;
    v47 += 8LL;
    if ( v46 == (_BYTE *)v47 )
      goto LABEL_67;
  }
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  sub_C7D6A0(*((__int64 *)&v54 + 1), 8LL * DWORD2(v55), 8);
  return v33;
}
