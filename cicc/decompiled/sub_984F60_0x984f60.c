// Function: sub_984F60
// Address: 0x984f60
//
__int64 __fastcall sub_984F60(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // r8
  unsigned int v8; // eax
  __int64 v9; // rcx
  char *v10; // r14
  char **v11; // rax
  char **v12; // rdx
  unsigned int v13; // r12d
  __int64 i; // r15
  char **v16; // rax
  unsigned __int8 v17; // r15
  char **v18; // rax
  char **v19; // rdx
  __int64 v20; // rax
  char *v21; // rdx
  char *v22; // [rsp+8h] [rbp-288h] BYREF
  _QWORD *v23; // [rsp+10h] [rbp-280h] BYREF
  __int64 v24; // [rsp+18h] [rbp-278h]
  _QWORD v25[16]; // [rsp+20h] [rbp-270h] BYREF
  __int64 v26; // [rsp+A0h] [rbp-1F0h] BYREF
  char **v27; // [rsp+A8h] [rbp-1E8h]
  __int64 v28; // [rsp+B0h] [rbp-1E0h]
  int v29; // [rsp+B8h] [rbp-1D8h]
  char v30; // [rsp+BCh] [rbp-1D4h]
  char v31; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v32; // [rsp+140h] [rbp-150h] BYREF
  char **v33; // [rsp+148h] [rbp-148h]
  __int64 v34; // [rsp+150h] [rbp-140h]
  int v35; // [rsp+158h] [rbp-138h]
  char v36; // [rsp+15Ch] [rbp-134h]
  char v37; // [rsp+160h] [rbp-130h] BYREF

  v23 = v25;
  v24 = 0x1000000001LL;
  v33 = (char **)&v37;
  v27 = (char **)&v31;
  v3 = *(_DWORD *)(a1 + 4);
  v22 = (char *)a2;
  v25[0] = a1;
  v32 = 0;
  v4 = 4LL * (v3 & 0x7FFFFFF);
  v34 = 32;
  v35 = 0;
  v36 = 1;
  v26 = 0;
  v28 = 16;
  v29 = 0;
  v30 = 1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v5 = *(_QWORD **)(a1 - 8);
    v6 = (__int64)&v5[v4];
  }
  else
  {
    v6 = a1;
    v5 = (_QWORD *)(a1 - v4 * 8);
  }
  v7 = sub_9841D0(v5, v6, (__int64 *)&v22);
  v8 = 1;
  if ( (_QWORD *)v6 != v7 )
  {
    v13 = 1;
    goto LABEL_15;
  }
  while ( 1 )
  {
    v9 = v8;
    v10 = (char *)v23[v8 - 1];
    LODWORD(v24) = v8 - 1;
    if ( !v36 )
      goto LABEL_18;
    v11 = v33;
    v6 = HIDWORD(v34);
    v12 = &v33[HIDWORD(v34)];
    if ( v33 != v12 )
    {
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_43;
      }
      goto LABEL_9;
    }
LABEL_43:
    if ( HIDWORD(v34) < (unsigned int)v34 )
    {
      v6 = (unsigned int)++HIDWORD(v34);
      *v12 = v10;
      ++v32;
    }
    else
    {
LABEL_18:
      v6 = (__int64)v10;
      sub_C8CC70(&v32, v10);
      if ( !(_BYTE)v12 )
        goto LABEL_9;
    }
    for ( i = *((_QWORD *)v10 + 2); i; i = *(_QWORD *)(i + 8) )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(i + 24);
        if ( !v30 )
          break;
        v16 = v27;
        v12 = &v27[HIDWORD(v28)];
        if ( v27 == v12 )
          goto LABEL_9;
        while ( (char *)v6 != *v16 )
        {
          if ( v12 == ++v16 )
            goto LABEL_9;
        }
        i = *(_QWORD *)(i + 8);
        if ( !i )
          goto LABEL_26;
      }
      if ( !sub_C8CA60(&v26, v6, v12, v9, v7) )
        goto LABEL_9;
    }
LABEL_26:
    if ( v22 == v10 )
      break;
    if ( (char *)a1 != v10 )
    {
      v17 = *v10;
      if ( (unsigned __int8)*v10 <= 0x1Cu || (unsigned __int8)sub_B46970(v10) || (unsigned int)v17 - 30 <= 0xA )
        goto LABEL_9;
    }
    if ( !v30 )
    {
LABEL_49:
      v6 = (__int64)v10;
      sub_C8CC70(&v26, v10);
      goto LABEL_36;
    }
    v18 = v27;
    v6 = HIDWORD(v28);
    v19 = &v27[HIDWORD(v28)];
    if ( v27 == v19 )
    {
LABEL_48:
      if ( HIDWORD(v28) >= (unsigned int)v28 )
        goto LABEL_49;
      v6 = (unsigned int)++HIDWORD(v28);
      *v19 = v10;
      ++v26;
    }
    else
    {
      while ( v10 != *v18 )
      {
        if ( v19 == ++v18 )
          goto LABEL_48;
      }
    }
LABEL_36:
    if ( (unsigned __int8)(*v10 - 22) > 6u )
    {
      v20 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
      v21 = &v10[-v20];
      if ( (v10[7] & 0x40) != 0 )
      {
        v21 = (char *)*((_QWORD *)v10 - 1);
        v10 = &v21[v20];
      }
      v6 = (__int64)&v23[(unsigned int)v24];
      sub_984620((__int64 *)&v23, (char *)v6, v21, v10);
      v8 = v24;
      goto LABEL_10;
    }
LABEL_9:
    v8 = v24;
LABEL_10:
    if ( !v8 )
    {
      v13 = 0;
      goto LABEL_12;
    }
  }
  v13 = 1;
LABEL_12:
  if ( v30 )
  {
    if ( !v36 )
      goto LABEL_14;
  }
  else
  {
    _libc_free(v27, v6);
    if ( !v36 )
LABEL_14:
      _libc_free(v33, v6);
  }
LABEL_15:
  if ( v23 != v25 )
    _libc_free(v23, v6);
  return v13;
}
