// Function: sub_DB5FD0
// Address: 0xdb5fd0
//
__int64 __fastcall sub_DB5FD0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdx
  unsigned int v6; // ecx
  unsigned int v7; // edi
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v11; // rax
  __int64 v12; // rbx
  char *v13; // r12
  _QWORD *v14; // rcx
  _BYTE *v15; // r8
  size_t v16; // r13
  _QWORD *v17; // rax
  char v18; // r13
  __int64 v19; // r9
  unsigned int v20; // r8d
  __int64 v21; // rdi
  int v22; // r11d
  __int64 *v23; // r10
  int v24; // edx
  int v25; // edx
  __int64 v26; // rax
  _QWORD *v27; // rdi
  int v28; // eax
  int v29; // r8d
  int v30; // r8d
  __int64 v31; // r10
  unsigned int v32; // ecx
  __int64 v33; // r9
  int v34; // edi
  __int64 *v35; // rsi
  int v36; // edi
  int v37; // edi
  __int64 v38; // r8
  int v39; // ecx
  __int64 *v40; // r9
  unsigned int v41; // ebx
  __int64 v42; // rsi
  int v43; // r9d
  __int64 v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+18h] [rbp-A8h]
  _BYTE *src; // [rsp+20h] [rbp-A0h]
  _QWORD *v48; // [rsp+28h] [rbp-98h]
  _QWORD *v49; // [rsp+28h] [rbp-98h]
  char v50; // [rsp+36h] [rbp-8Ah]
  char v51; // [rsp+37h] [rbp-89h]
  __int64 v52; // [rsp+38h] [rbp-88h]
  size_t v53; // [rsp+48h] [rbp-78h] BYREF
  _QWORD v54[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v55[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v56; // [rsp+70h] [rbp-50h]
  __int64 v57; // [rsp+78h] [rbp-48h]
  __int64 v58; // [rsp+80h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 896);
  v5 = *(_QWORD *)(a1 + 880);
  if ( !v4 )
  {
    v11 = *(_QWORD *)(a2 + 32);
    v45 = *(_QWORD *)(a2 + 40);
    if ( v11 == v45 )
    {
      v50 = 1;
      v19 = a1 + 872;
      v51 = 1;
      goto LABEL_55;
    }
    goto LABEL_6;
  }
  v6 = v4 - 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
LABEL_3:
    if ( v8 != (__int64 *)(v5 + 16LL * v4) )
      return *((unsigned __int16 *)v8 + 4);
    v11 = *(_QWORD *)(a2 + 32);
    v45 = *(_QWORD *)(a2 + 40);
    if ( v45 == v11 )
      goto LABEL_53;
LABEL_6:
    v46 = v11;
    v50 = 1;
    v51 = 1;
    while ( 1 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)v46 + 56LL);
      v52 = *(_QWORD *)v46 + 48LL;
      if ( v52 != v12 )
        break;
LABEL_28:
      v46 += 8;
      if ( v45 == v46 )
      {
        v4 = *(_DWORD *)(a1 + 896);
        v5 = *(_QWORD *)(a1 + 880);
        v19 = a1 + 872;
        if ( v4 )
        {
          v6 = v4 - 1;
          goto LABEL_31;
        }
LABEL_55:
        ++*(_QWORD *)(a1 + 872);
        v4 = 0;
        goto LABEL_56;
      }
    }
    while ( 1 )
    {
      v13 = (char *)(v12 - 24);
      v54[0] = v55;
      if ( !v12 )
        v13 = 0;
      v14 = *(_QWORD **)(*(_QWORD *)a1 + 40LL);
      v15 = (_BYTE *)v14[29];
      v16 = v14[30];
      if ( &v15[v16] && !v15 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v53 = v14[30];
      if ( v16 > 0xF )
      {
        src = v15;
        v48 = v14;
        v26 = sub_22409D0(v54, &v53, 0);
        v14 = v48;
        v15 = src;
        v54[0] = v26;
        v27 = (_QWORD *)v26;
        v55[0] = v53;
      }
      else
      {
        if ( v16 == 1 )
        {
          LOBYTE(v55[0]) = *v15;
          v17 = v55;
          goto LABEL_21;
        }
        if ( !v16 )
        {
          v17 = v55;
          goto LABEL_21;
        }
        v27 = v55;
      }
      v49 = v14;
      memcpy(v27, v15, v16);
      v16 = v53;
      v17 = (_QWORD *)v54[0];
      v14 = v49;
LABEL_21:
      v54[1] = v16;
      *((_BYTE *)v17 + v16) = 0;
      v56 = v14[33];
      v57 = v14[34];
      v58 = v14[35];
      if ( (unsigned int)(v56 - 42) > 1 )
      {
        if ( (_QWORD *)v54[0] != v55 )
          j_j___libc_free_0(v54[0], v55[0] + 1LL);
        v18 = sub_98CD80(v13);
        if ( v18 )
          v18 = v50;
        else
          v50 = 0;
      }
      else
      {
        v18 = v50;
        if ( (_QWORD *)v54[0] != v55 )
          j_j___libc_free_0(v54[0], v55[0] + 1LL);
      }
      if ( *v13 == 62 )
      {
        if ( sub_B46500((unsigned __int8 *)v13) || (v13[2] & 1) != 0 )
        {
LABEL_27:
          v51 = 0;
          if ( !v18 )
            goto LABEL_28;
          goto LABEL_13;
        }
      }
      else if ( (unsigned __int8)sub_B46790((unsigned __int8 *)v13, 0) || (unsigned __int8)sub_B46490((__int64)v13) )
      {
        goto LABEL_27;
      }
      if ( !v18 && !v51 )
        goto LABEL_28;
LABEL_13:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v52 == v12 )
        goto LABEL_28;
    }
  }
  v28 = 1;
  while ( v9 != -4096 )
  {
    v43 = v28 + 1;
    v7 = v6 & (v28 + v7);
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      goto LABEL_3;
    v28 = v43;
  }
  v11 = *(_QWORD *)(a2 + 32);
  v45 = *(_QWORD *)(a2 + 40);
  if ( v11 != v45 )
    goto LABEL_6;
LABEL_53:
  v50 = 1;
  v19 = a1 + 872;
  v51 = 1;
LABEL_31:
  v20 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v20);
  v21 = *v8;
  if ( *v8 == a2 )
    return *((unsigned __int16 *)v8 + 4);
  v22 = 1;
  v23 = 0;
  while ( v21 != -4096 )
  {
    if ( v21 == -8192 && !v23 )
      v23 = v8;
    v20 = v6 & (v22 + v20);
    v8 = (__int64 *)(v5 + 16LL * v20);
    v21 = *v8;
    if ( a2 == *v8 )
      return *((unsigned __int16 *)v8 + 4);
    ++v22;
  }
  v24 = *(_DWORD *)(a1 + 888);
  if ( v23 )
    v8 = v23;
  ++*(_QWORD *)(a1 + 872);
  v25 = v24 + 1;
  if ( 4 * v25 >= 3 * v4 )
  {
LABEL_56:
    sub_DB5DF0(v19, 2 * v4);
    v29 = *(_DWORD *)(a1 + 896);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 880);
      v25 = *(_DWORD *)(a1 + 888) + 1;
      v32 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v31 + 16LL * v32);
      v33 = *v8;
      if ( a2 != *v8 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v35 )
            v35 = v8;
          v32 = v30 & (v34 + v32);
          v8 = (__int64 *)(v31 + 16LL * v32);
          v33 = *v8;
          if ( a2 == *v8 )
            goto LABEL_38;
          ++v34;
        }
        if ( v35 )
          v8 = v35;
      }
      goto LABEL_38;
    }
    goto LABEL_90;
  }
  if ( v4 - (v25 + *(_DWORD *)(a1 + 892)) <= v4 >> 3 )
  {
    sub_DB5DF0(v19, v4);
    v36 = *(_DWORD *)(a1 + 896);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 880);
      v39 = 1;
      v40 = 0;
      v41 = v37 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = *(_DWORD *)(a1 + 888) + 1;
      v8 = (__int64 *)(v38 + 16LL * v41);
      v42 = *v8;
      if ( a2 != *v8 )
      {
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v40 )
            v40 = v8;
          v41 = v37 & (v39 + v41);
          v8 = (__int64 *)(v38 + 16LL * v41);
          v42 = *v8;
          if ( a2 == *v8 )
            goto LABEL_38;
          ++v39;
        }
        if ( v40 )
          v8 = v40;
      }
      goto LABEL_38;
    }
LABEL_90:
    ++*(_DWORD *)(a1 + 888);
    BUG();
  }
LABEL_38:
  *(_DWORD *)(a1 + 888) = v25;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 892);
  *v8 = a2;
  *((_BYTE *)v8 + 8) = v50;
  *((_BYTE *)v8 + 9) = v51;
  return *((unsigned __int16 *)v8 + 4);
}
