// Function: sub_14A7C10
// Address: 0x14a7c10
//
_BYTE *__fastcall sub_14A7C10(_BYTE *a1, _BYTE *a2)
{
  _BYTE *v2; // r12
  _BYTE *v3; // rbx
  _QWORD *v4; // rax
  int v5; // edx
  _QWORD *v6; // rcx
  unsigned int v7; // eax
  _BYTE *v8; // rsi
  __int64 v9; // rax
  _BYTE *v10; // rdx
  unsigned __int64 *v11; // rax
  int v12; // edx
  _QWORD *v13; // rcx
  unsigned int v14; // eax
  _BYTE *v15; // rsi
  int v16; // edi
  __int64 v17; // rax
  _BYTE *v18; // rdx
  int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // r9
  __int64 v23; // rcx
  _BYTE *v24; // rdi
  int v25; // eax
  int v26; // esi
  __int64 v27; // r12
  unsigned __int64 v28; // rdi
  int v30; // edi
  _BYTE *v31; // [rsp+8h] [rbp-F8h] BYREF
  __int64 v32; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v33; // [rsp+18h] [rbp-E8h]
  _QWORD *v34; // [rsp+20h] [rbp-E0h] BYREF
  int v35; // [rsp+28h] [rbp-D8h]
  _BYTE *v36; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+48h] [rbp-B8h]
  _BYTE v38[32]; // [rsp+50h] [rbp-B0h] BYREF
  _BYTE *v39; // [rsp+70h] [rbp-90h] BYREF
  __int64 v40; // [rsp+78h] [rbp-88h]
  _QWORD *v41; // [rsp+80h] [rbp-80h] BYREF
  int v42; // [rsp+88h] [rbp-78h]
  _BYTE *v43; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v44; // [rsp+A8h] [rbp-58h]
  _BYTE v45[80]; // [rsp+B0h] [rbp-50h] BYREF

  if ( !a1 )
    return 0;
  v2 = a2;
  if ( !a2 )
    return 0;
  v3 = a1;
  if ( a1 == a2 )
    return a1;
  v32 = 0;
  v33 = 1;
  v4 = &v34;
  do
    *v4++ = -8;
  while ( v4 != &v36 );
  v36 = v38;
  v37 = 0x400000000LL;
  while ( 1 )
  {
    if ( (v33 & 1) != 0 )
    {
      v5 = 3;
      v6 = &v34;
    }
    else
    {
      v6 = v34;
      v5 = v35 - 1;
      if ( !v35 )
        goto LABEL_12;
    }
    v7 = v5 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v8 = (_BYTE *)v6[v7];
    if ( v8 == v3 )
LABEL_10:
      sub_16BD130("Cycle found in TBAA metadata.", 1);
    v30 = 1;
    while ( v8 != (_BYTE *)-8LL )
    {
      v7 = v5 & (v30 + v7);
      v8 = (_BYTE *)v6[v7];
      if ( v8 == v3 )
        goto LABEL_10;
      ++v30;
    }
LABEL_12:
    v39 = v3;
    sub_14A7930((__int64)&v32, &v39);
    v9 = *((unsigned int *)v3 + 2);
    if ( (unsigned int)v9 <= 2 )
    {
      if ( (_DWORD)v9 != 2 )
        break;
      v9 = 2;
    }
    else
    {
      v10 = *(_BYTE **)&v3[-8 * v9];
      if ( (unsigned __int8)(*v10 - 4) <= 0x1Eu )
        goto LABEL_43;
    }
    v10 = *(_BYTE **)&v3[8 * (1 - v9)];
    if ( !v10 || (unsigned __int8)(*v10 - 4) > 0x1Eu )
      break;
LABEL_43:
    v3 = v10;
  }
  v39 = 0;
  v40 = 1;
  v11 = (unsigned __int64 *)&v41;
  do
    *v11++ = -8;
  while ( v11 != (unsigned __int64 *)&v43 );
  v43 = v45;
  v44 = 0x400000000LL;
  while ( 2 )
  {
    if ( (v40 & 1) != 0 )
    {
      v12 = 3;
      v13 = &v41;
      goto LABEL_21;
    }
    v13 = v41;
    v12 = v42 - 1;
    if ( v42 )
    {
LABEL_21:
      v14 = v12 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v15 = (_BYTE *)v13[v14];
      if ( v15 == v2 )
        goto LABEL_10;
      v16 = 1;
      while ( v15 != (_BYTE *)-8LL )
      {
        v14 = v12 & (v16 + v14);
        v15 = (_BYTE *)v13[v14];
        if ( v15 == v2 )
          goto LABEL_10;
        ++v16;
      }
    }
    v31 = v2;
    sub_14A7930((__int64)&v39, &v31);
    v17 = *((unsigned int *)v2 + 2);
    if ( (unsigned int)v17 > 2 )
    {
      v18 = *(_BYTE **)&v2[-8 * v17];
      if ( (unsigned __int8)(*v18 - 4) > 0x1Eu )
        goto LABEL_26;
      goto LABEL_44;
    }
    if ( (_DWORD)v17 != 2 )
      break;
    v17 = 2;
LABEL_26:
    v18 = *(_BYTE **)&v2[8 * (1 - v17)];
    if ( v18 && (unsigned __int8)(*v18 - 4) <= 0x1Eu )
    {
LABEL_44:
      v2 = v18;
      continue;
    }
    break;
  }
  v19 = v44 - 1;
  if ( (int)v37 - 1 < 0 || v19 < 0 )
  {
    v27 = 0;
  }
  else
  {
    v20 = (int)v37 - 2;
    v21 = -8 * v20 + 8LL * v19;
    v22 = &v36[8 * (int)v37 - 8 + -8 * v20];
    v23 = 0;
    v24 = &v43[v21];
    while ( 1 )
    {
      v27 = v23;
      v23 = *(_QWORD *)&v22[8 * v20];
      if ( v23 != *(_QWORD *)&v24[8 * v20] )
        break;
      v25 = v44 - v37 + v20;
      v26 = v20--;
      if ( v26 < 0 || v25 < 0 )
      {
        v27 = v23;
        if ( v43 != v45 )
          goto LABEL_34;
        goto LABEL_35;
      }
    }
  }
  if ( v43 != v45 )
LABEL_34:
    _libc_free((unsigned __int64)v43);
LABEL_35:
  if ( (v40 & 1) != 0 )
  {
    v28 = (unsigned __int64)v36;
    if ( v36 != v38 )
      goto LABEL_37;
  }
  else
  {
    j___libc_free_0(v41);
    v28 = (unsigned __int64)v36;
    if ( v36 != v38 )
LABEL_37:
      _libc_free(v28);
  }
  if ( (v33 & 1) == 0 )
    j___libc_free_0(v34);
  return (_BYTE *)v27;
}
