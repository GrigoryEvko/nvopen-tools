// Function: sub_DF54F0
// Address: 0xdf54f0
//
__int64 __fastcall sub_DF54F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 **v7; // rbx
  __int64 v8; // rdx
  __int64 **v9; // r12
  __int64 v10; // rcx
  __int64 *v11; // rdx
  unsigned __int8 v12; // al
  __int64 *v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // r14
  __int64 v17; // r15
  __int64 *v18; // r13
  unsigned int v19; // r12d
  __int64 *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-228h]
  __int64 v23; // [rsp+20h] [rbp-210h] BYREF
  __int64 *v24; // [rsp+28h] [rbp-208h]
  __int64 v25; // [rsp+30h] [rbp-200h]
  int v26; // [rsp+38h] [rbp-1F8h]
  unsigned __int8 v27; // [rsp+3Ch] [rbp-1F4h]
  char v28; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 v29; // [rsp+C0h] [rbp-170h] BYREF
  char *v30; // [rsp+C8h] [rbp-168h]
  __int64 v31; // [rsp+D0h] [rbp-160h]
  int v32; // [rsp+D8h] [rbp-158h]
  char v33; // [rsp+DCh] [rbp-154h]
  char v34; // [rsp+E0h] [rbp-150h] BYREF
  __int64 v35; // [rsp+160h] [rbp-D0h] BYREF
  char *v36; // [rsp+168h] [rbp-C8h]
  __int64 v37; // [rsp+170h] [rbp-C0h]
  int v38; // [rsp+178h] [rbp-B8h]
  char v39; // [rsp+17Ch] [rbp-B4h]
  char v40; // [rsp+180h] [rbp-B0h] BYREF

  v24 = (__int64 *)&v28;
  v6 = *(_BYTE *)(a2 - 16);
  v22 = a2;
  v23 = 0;
  v25 = 16;
  v26 = 0;
  v27 = 1;
  if ( (v6 & 2) != 0 )
  {
    v7 = *(__int64 ***)(a2 - 32);
    v8 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v7 = (__int64 **)(a2 - 8LL * ((v6 >> 2) & 0xF) - 16);
    v8 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  }
  v9 = &v7[v8];
  if ( v9 == v7 )
    goto LABEL_22;
  v10 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *v7;
        if ( (unsigned __int8)(*(_BYTE *)*v7 - 5) > 0x1Fu )
          goto LABEL_16;
        v12 = *((_BYTE *)v11 - 16);
        if ( (v12 & 2) != 0 )
          break;
        a2 = (*((_WORD *)v11 - 8) >> 6) & 0xF;
        if ( ((*((_WORD *)v11 - 8) >> 6) & 0xFu) > 1 )
        {
          v11 -= (v12 >> 2) & 0xF;
          v13 = v11 - 2;
          goto LABEL_9;
        }
LABEL_16:
        if ( v9 == ++v7 )
          goto LABEL_17;
      }
      if ( *((_DWORD *)v11 - 6) <= 1u )
        goto LABEL_16;
      v13 = (__int64 *)*(v11 - 4);
LABEL_9:
      a2 = v13[1];
      if ( !a2 || (unsigned __int8)(*(_BYTE *)a2 - 5) > 0x1Fu )
        goto LABEL_16;
      if ( (_BYTE)v10 )
        break;
LABEL_27:
      ++v7;
      sub_C8CC70((__int64)&v23, a2, (__int64)v11, v10, a5, a6);
      v10 = v27;
      if ( v9 == v7 )
        goto LABEL_17;
    }
    v14 = v24;
    v11 = &v24[HIDWORD(v25)];
    if ( v24 != v11 )
    {
      while ( a2 != *v14 )
      {
        if ( v11 == ++v14 )
          goto LABEL_26;
      }
      goto LABEL_16;
    }
LABEL_26:
    if ( HIDWORD(v25) >= (unsigned int)v25 )
      goto LABEL_27;
    ++v7;
    ++HIDWORD(v25);
    *v11 = a2;
    v10 = v27;
    ++v23;
  }
  while ( v9 != v7 );
LABEL_17:
  v15 = v24;
  if ( (_BYTE)v10 )
    v16 = &v24[HIDWORD(v25)];
  else
    v16 = &v24[(unsigned int)v25];
  if ( v24 == v16 )
    goto LABEL_22;
  while ( 1 )
  {
    v17 = *v15;
    v18 = v15;
    if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v16 == ++v15 )
      goto LABEL_22;
  }
  if ( v16 == v15 )
  {
LABEL_22:
    v19 = 1;
    if ( !v27 )
      goto LABEL_49;
    return v19;
  }
  while ( 2 )
  {
    a2 = v17;
    v29 = 0;
    v30 = &v34;
    v31 = 16;
    v32 = 0;
    v33 = 1;
    sub_DF4EC0(a1, v17, (__int64)&v29, v10, a5, a6);
    if ( HIDWORD(v31) == v32 )
    {
LABEL_34:
      if ( !v33 )
        _libc_free(v30, a2);
      v21 = v18 + 1;
      if ( v18 + 1 == v16 )
        goto LABEL_22;
      v17 = *v21;
      for ( ++v18; (unsigned __int64)*v21 >= 0xFFFFFFFFFFFFFFFELL; v18 = v21 )
      {
        if ( v16 == ++v21 )
          goto LABEL_22;
        v17 = *v21;
      }
      if ( v16 == v18 )
        goto LABEL_22;
      continue;
    }
    break;
  }
  v35 = 0;
  v36 = &v40;
  v37 = 16;
  v38 = 0;
  v39 = 1;
  sub_DF4EC0(v22, v17, (__int64)&v35, v10, a5, a6);
  a2 = (__int64)&v35;
  if ( !(unsigned __int8)sub_DF53E0((__int64)&v29, (__int64)&v35) )
  {
    if ( !v39 )
      _libc_free(v36, &v35);
    goto LABEL_34;
  }
  if ( !v39 )
  {
    _libc_free(v36, &v35);
    if ( v33 )
      goto LABEL_48;
LABEL_51:
    _libc_free(v30, &v35);
    goto LABEL_48;
  }
  if ( !v33 )
    goto LABEL_51;
LABEL_48:
  v19 = 0;
  if ( v27 )
    return v19;
LABEL_49:
  _libc_free(v24, a2);
  return v19;
}
