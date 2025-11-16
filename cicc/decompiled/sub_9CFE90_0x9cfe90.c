// Function: sub_9CFE90
// Address: 0x9cfe90
//
__int64 *__fastcall sub_9CFE90(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rcx
  __int64 *v6; // rsi
  char v7; // dl
  char v8; // dl
  __int64 v9; // rdx
  __int64 v10; // [rsp+10h] [rbp-290h] BYREF
  char v11; // [rsp+18h] [rbp-288h]
  __int64 v12; // [rsp+20h] [rbp-280h] BYREF
  char v13; // [rsp+28h] [rbp-278h]
  _QWORD v14[4]; // [rsp+30h] [rbp-270h] BYREF
  char v15; // [rsp+50h] [rbp-250h]
  char v16; // [rsp+51h] [rbp-24Fh]
  __int64 *v17; // [rsp+60h] [rbp-240h] BYREF
  __int64 v18; // [rsp+68h] [rbp-238h]
  _BYTE v19[560]; // [rsp+70h] [rbp-230h] BYREF

  v2 = a2 + 32;
  sub_A4DCE0(&v17, a2 + 32, 21, 0);
  if ( ((unsigned __int64)v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = (unsigned __int64)v17 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else if ( *(_QWORD *)(a2 + 1920) == *(_QWORD *)(a2 + 1912) )
  {
    v17 = (__int64 *)v19;
    v18 = 0x4000000000LL;
    while ( 1 )
    {
      v6 = (__int64 *)v2;
      sub_9CEFB0((__int64)&v10, v2, 0, v4);
      v7 = v11 & 1;
      v11 = (2 * (v11 & 1)) | v11 & 0xFD;
      if ( v7 )
      {
        v6 = &v10;
        sub_9C9090(a1, &v10);
        goto LABEL_24;
      }
      if ( (_DWORD)v10 == 1 )
      {
        *a1 = 1;
        goto LABEL_29;
      }
      if ( (v10 & 0xFFFFFFFD) == 0 )
      {
        v16 = 1;
        v6 = (__int64 *)(a2 + 8);
        v14[0] = "Malformed block";
        v15 = 3;
        sub_9C81F0(a1, a2 + 8, (__int64)v14);
        goto LABEL_24;
      }
      sub_A4B600(&v12, v2, HIDWORD(v10), &v17, 0);
      v8 = v13 & 1;
      v13 = (2 * (v13 & 1)) | v13 & 0xFD;
      if ( v8 )
        break;
      if ( (_DWORD)v12 != 1 )
        goto LABEL_37;
      v9 = *(_QWORD *)(a2 + 1920);
      if ( v9 == *(_QWORD *)(a2 + 1928) )
      {
        sub_9CBC60((const __m128i **)(a2 + 1912), *(const __m128i **)(a2 + 1920));
        v9 = *(_QWORD *)(a2 + 1920) - 32LL;
      }
      else
      {
        if ( v9 )
        {
          *(_QWORD *)(v9 + 8) = 0;
          *(_QWORD *)v9 = v9 + 16;
          *(_BYTE *)(v9 + 16) = 0;
          v9 = *(_QWORD *)(a2 + 1920);
        }
        *(_QWORD *)(a2 + 1920) = v9 + 32;
      }
      if ( (unsigned __int8)sub_9C57B0(v17, (unsigned int)v18, (__int64 *)v9) )
      {
LABEL_37:
        v16 = 1;
        v6 = (__int64 *)(a2 + 8);
        v14[0] = "Invalid operand bundle record";
        v15 = 3;
        sub_9C81F0(a1, a2 + 8, (__int64)v14);
        goto LABEL_38;
      }
      LODWORD(v18) = 0;
      if ( (v13 & 2) != 0 )
        goto LABEL_35;
      if ( (v13 & 1) != 0 && v12 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
      if ( (v11 & 2) != 0 )
        goto LABEL_33;
      if ( (v11 & 1) != 0 && v10 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
    }
    v6 = &v12;
    sub_9C8CD0(a1, &v12);
LABEL_38:
    if ( (v13 & 2) != 0 )
LABEL_35:
      sub_9CE230(&v12);
    if ( (v13 & 1) != 0 && v12 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
LABEL_24:
    if ( (v11 & 2) != 0 )
LABEL_33:
      sub_9CEF10(&v10);
    if ( (v11 & 1) != 0 && v10 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
LABEL_29:
    if ( v17 != (__int64 *)v19 )
      _libc_free(v17, v6);
  }
  else
  {
    v19[17] = 1;
    v17 = (__int64 *)"Invalid multiple blocks";
    v19[16] = 3;
    sub_9C81F0(a1, a2 + 8, (__int64)&v17);
  }
  return a1;
}
