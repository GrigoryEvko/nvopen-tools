// Function: sub_9CF9C0
// Address: 0x9cf9c0
//
__int64 *__fastcall sub_9CF9C0(__int64 *a1, _QWORD *a2)
{
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 *v6; // rsi
  char v7; // dl
  char v8; // al
  char v9; // dl
  __int64 v10; // rdx
  unsigned __int64 v11; // r11
  __int64 v12; // r9
  const char *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  char v16; // al
  __int64 v17; // rdx
  const char *v18; // rdi
  const char *v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // [rsp+8h] [rbp-2B8h]
  __int64 v23; // [rsp+10h] [rbp-2B0h]
  char v24; // [rsp+18h] [rbp-2A8h]
  __int64 v25; // [rsp+30h] [rbp-290h] BYREF
  char v26; // [rsp+38h] [rbp-288h]
  __int64 v27; // [rsp+40h] [rbp-280h] BYREF
  char v28; // [rsp+48h] [rbp-278h]
  const char *v29; // [rsp+50h] [rbp-270h] BYREF
  __int64 v30; // [rsp+58h] [rbp-268h]
  __int64 v31; // [rsp+60h] [rbp-260h]
  _BYTE v32[8]; // [rsp+68h] [rbp-258h] BYREF
  char v33; // [rsp+70h] [rbp-250h]
  char v34; // [rsp+71h] [rbp-24Fh]
  unsigned __int64 v35; // [rsp+80h] [rbp-240h] BYREF
  __int64 v36; // [rsp+88h] [rbp-238h]
  _BYTE v37[560]; // [rsp+90h] [rbp-230h] BYREF

  v4 = (__int64)(a2 + 4);
  sub_A4DCE0(&v35, a2 + 4, 26, 0);
  if ( (v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v35 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  if ( a2[243] )
  {
    v37[17] = 1;
    v35 = (unsigned __int64)"Invalid multiple synchronization scope names blocks";
    v37[16] = 3;
    sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v35);
    return a1;
  }
  v35 = (unsigned __int64)v37;
  v36 = 0x4000000000LL;
  while ( 1 )
  {
    v6 = (__int64 *)v4;
    sub_9CEFB0((__int64)&v25, v4, 0, v5);
    v7 = v26 & 1;
    v8 = (2 * (v26 & 1)) | v26 & 0xFD;
    v26 = v8;
    if ( v7 )
    {
      v26 = v8 & 0xFD;
      v21 = v25;
      v25 = 0;
      *a1 = v21 | 1;
      goto LABEL_36;
    }
    if ( (_DWORD)v25 == 1 )
    {
      if ( a2[243] )
      {
        *a1 = 1;
        goto LABEL_31;
      }
      v34 = 1;
      v20 = "Invalid empty synchronization scope names block";
      goto LABEL_28;
    }
    if ( (v25 & 0xFFFFFFFD) == 0 )
    {
      v34 = 1;
      v20 = "Malformed block";
LABEL_28:
      v6 = a2 + 1;
      v29 = v20;
      v33 = 3;
      sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v29);
      goto LABEL_29;
    }
    sub_A4B600(&v27, v4, HIDWORD(v25), &v35, 0);
    v9 = v28 & 1;
    v28 = (2 * (v28 & 1)) | v28 & 0xFD;
    if ( v9 )
    {
      v6 = &v27;
      sub_9C8CD0(a1, &v27);
      goto LABEL_40;
    }
    if ( (_DWORD)v27 != 1 )
      break;
    v10 = (unsigned int)v36;
    v30 = 0;
    v29 = v32;
    v11 = v35;
    v31 = 16;
    v12 = (unsigned int)v36;
    if ( (unsigned int)v36 > 0x10uLL )
    {
      v22 = v35;
      v23 = (unsigned int)v36;
      sub_C8D290(&v29, v32, (unsigned int)v36, 1);
      v10 = v23;
      v11 = v22;
      v13 = &v29[v30];
    }
    else
    {
      v13 = v32;
      if ( !(8LL * (unsigned int)v36) )
        goto LABEL_14;
    }
    v14 = 0;
    do
    {
      v13[v14] = *(_QWORD *)(v11 + 8 * v14);
      ++v14;
    }
    while ( v10 != v14 );
    v13 = v29;
    v12 = v30 + v10;
LABEL_14:
    v15 = a2[54];
    v30 = v12;
    v16 = sub_B6F810(v15, v13, v12);
    v17 = a2[243];
    if ( (unsigned __int64)(v17 + 1) > a2[244] )
    {
      v13 = (const char *)(a2 + 245);
      v24 = v16;
      sub_C8D290(a2 + 242, a2 + 245, v17 + 1, 1);
      v17 = a2[243];
      v16 = v24;
    }
    v5 = a2[242];
    *(_BYTE *)(v5 + v17) = v16;
    v18 = v29;
    ++a2[243];
    LODWORD(v36) = 0;
    if ( v18 != v32 )
      _libc_free(v18, v13);
    if ( (v28 & 2) != 0 )
      goto LABEL_48;
    if ( (v28 & 1) != 0 && v27 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
    if ( (v26 & 2) != 0 )
      goto LABEL_46;
    if ( (v26 & 1) != 0 )
    {
      if ( v25 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
    }
  }
  v34 = 1;
  v6 = a2 + 1;
  v29 = "Invalid sync scope record";
  v33 = 3;
  sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v29);
LABEL_40:
  if ( (v28 & 2) != 0 )
LABEL_48:
    sub_9CE230(&v27);
  if ( (v28 & 1) != 0 && v27 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
LABEL_29:
  if ( (v26 & 2) != 0 )
LABEL_46:
    sub_9CEF10(&v25);
  if ( (v26 & 1) != 0 )
  {
LABEL_36:
    if ( v25 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
  }
LABEL_31:
  if ( (_BYTE *)v35 != v37 )
    _libc_free(v35, v6);
  return a1;
}
