// Function: sub_D31D20
// Address: 0xd31d20
//
__int64 __fastcall sub_D31D20(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  unsigned int v5; // eax
  __int64 v6; // r8
  unsigned __int64 v7; // r9
  __int64 v8; // rax
  int v9; // r14d
  __int64 v10; // rcx
  unsigned int v11; // edx
  _QWORD *i; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rbx
  char *v16; // rax
  __int64 v17; // rsi
  char v18; // dl
  char v19; // al
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // [rsp+8h] [rbp-D8h]
  _QWORD *v24; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+18h] [rbp-C8h]
  _QWORD v26[6]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v27; // [rsp+50h] [rbp-90h] BYREF
  char *v28; // [rsp+58h] [rbp-88h]
  __int64 v29; // [rsp+60h] [rbp-80h]
  int v30; // [rsp+68h] [rbp-78h]
  unsigned __int8 v31; // [rsp+6Ch] [rbp-74h]
  char v32; // [rsp+70h] [rbp-70h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 14 )
    return 1;
  LOBYTE(v5) = sub_D30750(*(unsigned __int8 **)a1, (unsigned __int8 *)a2, a3);
  v3 = v5;
  if ( (_BYTE)v5 )
    return 1;
  v8 = *(_QWORD *)(a1 + 24);
  v31 = 1;
  v9 = 39;
  v24 = v26;
  v10 = 1;
  v11 = 1;
  v26[0] = v8;
  v27 = 0;
  v29 = 8;
  v30 = 0;
  v25 = 0x600000001LL;
  v28 = &v32;
  for ( i = v26; ; i = v24 )
  {
    v13 = v11;
    v14 = v11 - 1;
    v15 = i[v13 - 1];
    LODWORD(v25) = v14;
    if ( !(_BYTE)v10 )
      goto LABEL_15;
    v16 = v28;
    v17 = HIDWORD(v29);
    v14 = (__int64)&v28[8 * HIDWORD(v29)];
    if ( v28 != (char *)v14 )
    {
      while ( v15 != *(_QWORD *)v16 )
      {
        v16 += 8;
        if ( (char *)v14 == v16 )
          goto LABEL_21;
      }
LABEL_11:
      v11 = v25;
      goto LABEL_12;
    }
LABEL_21:
    if ( HIDWORD(v29) < (unsigned int)v29 )
    {
      v17 = (unsigned int)++HIDWORD(v29);
      *(_QWORD *)v14 = v15;
      v10 = v31;
      ++v27;
      v19 = *(_BYTE *)v15;
      if ( *(_BYTE *)v15 <= 0x1Cu )
        goto LABEL_17;
    }
    else
    {
LABEL_15:
      v17 = v15;
      sub_C8CC70((__int64)&v27, v15, v14, v10, v6, v7);
      v10 = v31;
      if ( !v18 )
        goto LABEL_11;
      v19 = *(_BYTE *)v15;
      if ( *(_BYTE *)v15 <= 0x1Cu )
        goto LABEL_17;
    }
    LOBYTE(v17) = v19 == 82 || v19 == 76;
    if ( (_BYTE)v17 )
      goto LABEL_11;
    if ( (v19 & 0xFD) != 0x54 )
    {
      v3 = 0;
      goto LABEL_17;
    }
    v20 = *(_QWORD *)(v15 + 16);
    v17 = HIDWORD(v25);
    v11 = v25;
    if ( v20 )
    {
      v21 = v20;
      v6 = 0;
      do
      {
        v21 = *(_QWORD *)(v21 + 8);
        ++v6;
      }
      while ( v21 );
      v17 = HIDWORD(v25);
      v7 = (unsigned int)v25 + v6;
      if ( HIDWORD(v25) < v7 )
      {
        v17 = (__int64)v26;
        v23 = v6;
        sub_C8D5F0((__int64)&v24, v26, (unsigned int)v25 + v6, 8u, v6, v7);
        v6 = v23;
      }
      v22 = &v24[(unsigned int)v25];
      do
      {
        *v22++ = *(_QWORD *)(v20 + 24);
        v20 = *(_QWORD *)(v20 + 8);
      }
      while ( v20 );
      v10 = v31;
      v11 = v6 + v25;
    }
    else if ( HIDWORD(v25) < (unsigned int)v25 )
    {
      v17 = (__int64)v26;
      sub_C8D5F0((__int64)&v24, v26, (unsigned int)v25, 8u, v6, v7);
      v11 = v25;
      v10 = v31;
    }
    LODWORD(v25) = v11;
LABEL_12:
    if ( !v11 )
      break;
    if ( !--v9 )
      goto LABEL_17;
  }
  v3 = 1;
LABEL_17:
  if ( !(_BYTE)v10 )
    _libc_free(v28, v17);
  if ( v24 != v26 )
    _libc_free(v24, v17);
  return v3;
}
