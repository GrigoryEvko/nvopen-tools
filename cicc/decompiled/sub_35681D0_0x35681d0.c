// Function: sub_35681D0
// Address: 0x35681d0
//
bool __fastcall sub_35681D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  unsigned int v7; // ecx
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 *v10; // r13
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  bool result; // al
  __int64 *v15; // r8
  __int64 *v16; // r13
  __int64 v17; // r14
  __int64 v18; // r15
  char v19; // al
  __int64 v20; // rax
  unsigned int v21; // ecx
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 *v25; // r14
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 *v28; // rbx
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 *v32; // r10
  int v33; // edx
  unsigned int v34; // ecx
  __int64 *v35; // r9
  __int64 v36; // r11
  int v37; // r9d
  int v38; // r8d
  int v39; // esi
  int v40; // edi
  __int64 *v41; // [rsp+8h] [rbp-38h]
  __int64 *v42; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 24);
  v7 = *(_DWORD *)(v6 + 224);
  v8 = *(_QWORD *)(v6 + 208);
  if ( v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v8 + 56LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_3;
    v39 = 1;
    while ( v11 != -4096 )
    {
      v9 = (v7 - 1) & (v39 + v9);
      v10 = (__int64 *)(v8 + 56LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      ++v39;
    }
  }
  v10 = (__int64 *)(v8 + 56LL * v7);
LABEL_3:
  if ( !(unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 8), a2, a3) )
  {
    v12 = (_QWORD *)v10[5];
    v13 = &v12[*((unsigned int *)v10 + 12)];
    if ( v12 == v13 )
      return 1;
    while ( a3 == *v12 || a2 == *v12 )
    {
      if ( v13 == ++v12 )
        return 1;
    }
    return 0;
  }
  v20 = *(_QWORD *)(a1 + 24);
  v21 = *(_DWORD *)(v20 + 224);
  v22 = *(_QWORD *)(v20 + 208);
  if ( !v21 )
    goto LABEL_35;
  v23 = (v21 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v22 + 56LL * v23);
  v24 = *v15;
  if ( a3 != *v15 )
  {
    v40 = 1;
    while ( v24 != -4096 )
    {
      v23 = (v21 - 1) & (v23 + v40);
      v15 = (__int64 *)(v22 + 56LL * v23);
      v24 = *v15;
      if ( a3 == *v15 )
        goto LABEL_20;
      ++v40;
    }
LABEL_35:
    v15 = (__int64 *)(v22 + 56LL * v21);
  }
LABEL_20:
  v25 = (__int64 *)v10[5];
  v26 = *((unsigned int *)v10 + 12);
  if ( &v25[v26] == v25 )
  {
LABEL_12:
    v16 = (__int64 *)v15[5];
    v41 = &v16[*((unsigned int *)v15 + 12)];
    if ( v41 == v16 )
      return 1;
    v17 = a1;
    while ( 1 )
    {
      v18 = *v16;
      sub_2E6D2E0(*(_QWORD *)(v17 + 8), a2, *v16);
      if ( a3 != v18 )
      {
        if ( v19 )
          break;
      }
      if ( v41 == ++v16 )
        return 1;
    }
    return 0;
  }
  v42 = &v25[v26];
  v27 = a3;
  v28 = v15;
  while ( 1 )
  {
    v29 = *v25;
    result = *v25 == a2 || *v25 == v27;
    if ( !result )
      break;
LABEL_24:
    if ( v42 == ++v25 )
    {
      v15 = v28;
      a3 = v27;
      goto LABEL_12;
    }
  }
  v30 = *((unsigned int *)v28 + 8);
  v31 = v28[2];
  v32 = (__int64 *)(v31 + 8 * v30);
  if ( !(_DWORD)v30 )
    return result;
  v33 = v30 - 1;
  v34 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
  v35 = (__int64 *)(v31 + 8LL * v34);
  v36 = *v35;
  if ( v29 == *v35 )
  {
LABEL_22:
    if ( v32 == v35 || !(unsigned __int8)sub_3568140(a1, v29, a2, v27) )
      return 0;
    goto LABEL_24;
  }
  v37 = 1;
  while ( v36 != -4096 )
  {
    v38 = v37 + 1;
    v34 = v33 & (v37 + v34);
    v35 = (__int64 *)(v31 + 8LL * v34);
    v36 = *v35;
    if ( v29 == *v35 )
      goto LABEL_22;
    v37 = v38;
  }
  return result;
}
