// Function: sub_98B9F0
// Address: 0x98b9f0
//
unsigned __int8 *__fastcall sub_98B9F0(unsigned __int8 *a1)
{
  __int64 v1; // rsi
  unsigned __int8 *v2; // rbx
  unsigned __int8 *v3; // r12
  unsigned __int8 *v4; // r15
  unsigned __int8 **v5; // rax
  unsigned __int8 **v6; // rdx
  unsigned int v7; // eax
  _QWORD *v8; // rdi
  unsigned __int8 *v9; // rdi
  char v10; // dl
  unsigned __int8 v11; // al
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rdx
  char *v19; // r8
  unsigned __int8 *v20; // rcx
  __int64 v21; // [rsp+8h] [rbp-E8h]
  _QWORD *v22; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v23; // [rsp+18h] [rbp-D8h]
  _QWORD v24[8]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int8 **v26; // [rsp+68h] [rbp-88h]
  __int64 v27; // [rsp+70h] [rbp-80h]
  int v28; // [rsp+78h] [rbp-78h]
  char v29; // [rsp+7Ch] [rbp-74h]
  char v30; // [rsp+80h] [rbp-70h] BYREF

  v1 = 6;
  v2 = 0;
  v26 = (unsigned __int8 **)&v30;
  v25 = 0;
  v27 = 8;
  v28 = 0;
  v29 = 1;
  v22 = v24;
  v24[0] = a1;
  v23 = 0x800000000LL;
  v3 = sub_98ACB0(a1, 6u);
  v4 = v3;
  if ( !v29 )
    goto LABEL_9;
LABEL_2:
  v5 = v26;
  v6 = &v26[HIDWORD(v27)];
  if ( v26 == v6 )
  {
LABEL_22:
    if ( HIDWORD(v27) >= (unsigned int)v27 )
      goto LABEL_9;
    ++HIDWORD(v27);
    *v6 = v4;
    ++v25;
LABEL_10:
    if ( HIDWORD(v27) - v28 == 8 )
      goto LABEL_16;
    v11 = *v4;
    if ( *v4 > 0x1Cu )
    {
      if ( v11 == 86 )
      {
        v13 = (unsigned int)v23;
        v14 = *((_QWORD *)v4 - 8);
        v15 = (unsigned int)v23 + 1LL;
        if ( v15 > HIDWORD(v23) )
        {
          v1 = (__int64)v24;
          v21 = *((_QWORD *)v4 - 8);
          sub_C8D5F0(&v22, v24, v15, 8);
          v13 = (unsigned int)v23;
          v14 = v21;
        }
        v22[v13] = v14;
        LODWORD(v23) = v23 + 1;
        v16 = (unsigned int)v23;
        v17 = *((_QWORD *)v4 - 4);
        if ( (unsigned __int64)(unsigned int)v23 + 1 > HIDWORD(v23) )
        {
          v1 = (__int64)v24;
          sub_C8D5F0(&v22, v24, (unsigned int)v23 + 1LL, 8);
          v16 = (unsigned int)v23;
        }
        v22[v16] = v17;
        v7 = v23 + 1;
        LODWORD(v23) = v23 + 1;
        goto LABEL_7;
      }
      if ( v11 == 84 )
      {
        v18 = 32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF);
        if ( (v4[7] & 0x40) != 0 )
        {
          v19 = (char *)*((_QWORD *)v4 - 1);
          v20 = (unsigned __int8 *)&v19[v18];
        }
        else
        {
          v20 = v4;
          v19 = (char *)&v4[-v18];
        }
        v1 = (__int64)&v22[(unsigned int)v23];
        sub_984620((__int64 *)&v22, (char *)v1, v19, v20);
        v7 = v23;
        goto LABEL_7;
      }
    }
    if ( v2 )
    {
      if ( v4 == v2 )
        goto LABEL_6;
LABEL_16:
      v8 = v22;
      goto LABEL_17;
    }
    v7 = v23;
    v8 = v22;
    v2 = v4;
    if ( (_DWORD)v23 )
      goto LABEL_8;
  }
  else
  {
    while ( v4 != *v5 )
    {
      if ( v6 == ++v5 )
        goto LABEL_22;
    }
LABEL_6:
    while ( 1 )
    {
      v7 = v23;
LABEL_7:
      v8 = v22;
      if ( !v7 )
        break;
LABEL_8:
      v1 = 6;
      v9 = (unsigned __int8 *)v8[v7 - 1];
      LODWORD(v23) = v7 - 1;
      v4 = sub_98ACB0(v9, 6u);
      if ( v29 )
        goto LABEL_2;
LABEL_9:
      v1 = (__int64)v4;
      sub_C8CC70(&v25, v4);
      if ( v10 )
        goto LABEL_10;
    }
  }
  if ( v2 )
    v3 = v2;
LABEL_17:
  if ( v8 != v24 )
    _libc_free(v8, v1);
  if ( !v29 )
    _libc_free(v26, v1);
  return v3;
}
