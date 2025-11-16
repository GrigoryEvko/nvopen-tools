// Function: sub_1612E30
// Address: 0x1612e30
//
_QWORD *__fastcall sub_1612E30(_QWORD *a1)
{
  __int64 v1; // r15
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // r9
  __int64 v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r12
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rsi
  unsigned int v18; // eax
  int v19; // edx
  _QWORD *v20; // r10
  __int64 v21; // rdi
  int v22; // r8d
  int v23; // r11d
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rbx
  _QWORD *v29; // rax
  _QWORD *v30; // r10
  _QWORD *v31; // rax
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  int v35; // edi
  __int64 v36; // r9
  __int64 v37; // rcx
  _QWORD *v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]
  unsigned int v40; // [rsp+18h] [rbp-38h]

  v1 = qword_4F9E390;
  if ( !qword_4F9E390 )
    return 0;
  v3 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 120LL))(a1);
  if ( v3 )
    return 0;
  if ( !qword_4F9E3A0 )
    sub_16C1EA0(&qword_4F9E3A0, sub_160CFB0, sub_160D0B0);
  v4 = qword_4F9E3A0;
  if ( !(unsigned __int8)((__int64 (*)(void))sub_16D5D40)() )
  {
    ++*(_DWORD *)(v4 + 8);
    v5 = *(unsigned int *)(v1 + 24);
    v6 = a1[2];
    if ( (_DWORD)v5 )
      goto LABEL_7;
LABEL_14:
    ++*(_QWORD *)v1;
    goto LABEL_15;
  }
  sub_16C30C0(v4);
  v5 = *(unsigned int *)(v1 + 24);
  v6 = a1[2];
  if ( !(_DWORD)v5 )
    goto LABEL_14;
LABEL_7:
  v7 = (unsigned int)(v5 - 1);
  v8 = *(_QWORD *)(v1 + 8);
  v9 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
  v10 = (unsigned int)v7 & (unsigned int)v9;
  v11 = (_QWORD *)(v8 + 16 * v10);
  v12 = *v11;
  if ( v6 != *v11 )
  {
    v20 = 0;
    v23 = 1;
    while ( v12 != -4 )
    {
      if ( v12 == -8 && !v20 )
        v20 = v11;
      v10 = (unsigned int)v7 & (v23 + (_DWORD)v10);
      v11 = (_QWORD *)(v8 + 16LL * (unsigned int)v10);
      v12 = *v11;
      if ( v6 == *v11 )
        goto LABEL_8;
      ++v23;
    }
    if ( !v20 )
      v20 = v11;
    v24 = *(_DWORD *)(v1 + 16);
    ++*(_QWORD *)v1;
    v19 = v24 + 1;
    if ( 4 * (v24 + 1) < (unsigned int)(3 * v5) )
    {
      if ( (int)v5 - *(_DWORD *)(v1 + 20) - v19 > (unsigned int)v5 >> 3 )
        goto LABEL_29;
      v40 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
      sub_1612C70(v1, v5);
      v32 = *(_DWORD *)(v1 + 24);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = *(_QWORD *)(v1 + 8);
        v35 = 1;
        LODWORD(v36) = v33 & v40;
        v19 = *(_DWORD *)(v1 + 16) + 1;
        v20 = (_QWORD *)(v34 + 16LL * (v33 & v40));
        v37 = *v20;
        if ( v6 == *v20 )
          goto LABEL_29;
        while ( v37 != -4 )
        {
          if ( !v3 && v37 == -8 )
            v3 = (__int64)v20;
          v36 = v33 & (unsigned int)(v36 + v35);
          v20 = (_QWORD *)(v34 + 16 * v36);
          v37 = *v20;
          if ( v6 == *v20 )
            goto LABEL_29;
          ++v35;
        }
        goto LABEL_19;
      }
      goto LABEL_57;
    }
LABEL_15:
    sub_1612C70(v1, 2 * v5);
    v15 = *(_DWORD *)(v1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(v1 + 8);
      v18 = (v15 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v19 = *(_DWORD *)(v1 + 16) + 1;
      v20 = (_QWORD *)(v17 + 16LL * v18);
      v21 = *v20;
      if ( v6 == *v20 )
      {
LABEL_29:
        *(_DWORD *)(v1 + 16) = v19;
        if ( *v20 != -4 )
          --*(_DWORD *)(v1 + 20);
        *v20 = v6;
        v20[1] = 0;
        goto LABEL_32;
      }
      v22 = 1;
      while ( v21 != -4 )
      {
        if ( !v3 && v21 == -8 )
          v3 = (__int64)v20;
        v18 = v16 & (v22 + v18);
        v20 = (_QWORD *)(v17 + 16LL * v18);
        v21 = *v20;
        if ( v6 == *v20 )
          goto LABEL_29;
        ++v22;
      }
LABEL_19:
      if ( v3 )
        v20 = (_QWORD *)v3;
      goto LABEL_29;
    }
LABEL_57:
    ++*(_DWORD *)(v1 + 16);
    BUG();
  }
LABEL_8:
  v13 = (_QWORD *)v11[1];
  if ( !v13 )
  {
    v20 = v11;
LABEL_32:
    v38 = v20;
    v39 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 16LL))(a1);
    v26 = v25;
    v27 = sub_1636970(a1[2]);
    if ( !v27 || (v28 = *(_QWORD *)(v27 + 24), v5 = *(_QWORD *)(v27 + 16), !v28) )
    {
      v5 = v39;
      v28 = v26;
    }
    v8 = 160;
    v29 = (_QWORD *)sub_22077B0(160);
    v30 = v38;
    v13 = v29;
    if ( v29 )
    {
      *v29 = 0;
      v31 = v29 + 10;
      *(v31 - 9) = 0;
      *(v31 - 8) = 0;
      v8 = (__int64)v13;
      *(v31 - 7) = 0;
      *(v31 - 6) = 0;
      *(v31 - 5) = 0;
      *(v31 - 4) = 0;
      *(v31 - 3) = 0;
      v13[8] = v31;
      v13[9] = 0;
      *((_BYTE *)v13 + 80) = 0;
      v13[12] = v13 + 14;
      v13[13] = 0;
      *((_BYTE *)v13 + 112) = 0;
      v13[17] = 0;
      sub_16D8060(v13, v5, v28, v39, v26, v1 + 32);
      v30 = v38;
    }
    v30[1] = v13;
  }
  if ( (unsigned __int8)sub_16D5D40(v8, v5, v12, v10, v7, v9) )
    sub_16C30E0(v4);
  else
    --*(_DWORD *)(v4 + 8);
  return v13;
}
