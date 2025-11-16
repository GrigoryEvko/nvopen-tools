// Function: sub_9CF210
// Address: 0x9cf210
//
__int64 __fastcall sub_9CF210(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned __int64 v4; // rbx
  int v5; // edx
  char v6; // al
  char v7; // dl
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // [rsp+18h] [rbp-A8h]
  __int64 v13; // [rsp+28h] [rbp-98h] BYREF
  __int64 v14; // [rsp+30h] [rbp-90h] BYREF
  char v15; // [rsp+38h] [rbp-88h]
  __int64 v16; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v17; // [rsp+48h] [rbp-78h]
  __int64 v18; // [rsp+50h] [rbp-70h] BYREF
  char v19; // [rsp+58h] [rbp-68h]
  __int64 v20[2]; // [rsp+60h] [rbp-60h] BYREF
  _BYTE v21[80]; // [rsp+70h] [rbp-50h] BYREF

  sub_A4DCE0(v20, a2, a3, 0);
  v4 = v20[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v20[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 16) |= 3u;
    *(_QWORD *)a1 = v4;
    return a1;
  }
  v12 = 0;
  while ( 1 )
  {
    sub_9CEA50((__int64)&v14, a2, 0, v3);
    v5 = v15 & 1;
    v3 = (unsigned int)(2 * v5);
    v6 = (2 * v5) | v15 & 0xFD;
    v15 = v6;
    if ( (_BYTE)v5 )
    {
      *(_BYTE *)(a1 + 16) |= 3u;
      v15 = v6 & 0xFD;
      v10 = v14;
      v14 = 0;
      *(_QWORD *)a1 = v10 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_9;
    }
    if ( (_DWORD)v14 == 2 )
    {
      sub_9CE5C0(v20, a2, 2, v3);
      v9 = v20[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v20[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *(_BYTE *)(a1 + 16) |= 3u;
LABEL_24:
        *(_QWORD *)a1 = v9;
        goto LABEL_25;
      }
      goto LABEL_17;
    }
    if ( (unsigned int)v14 <= 2 )
    {
      if ( (_DWORD)v14 )
      {
        v7 = *(_BYTE *)(a1 + 16);
        *(_QWORD *)(a1 + 8) = v4;
        *(_QWORD *)a1 = v12;
        *(_BYTE *)(a1 + 16) = v7 & 0xFC | 2;
        goto LABEL_8;
      }
      v21[17] = 1;
      v20[0] = (__int64)"Malformed block";
      v21[16] = 3;
      sub_9C8190(&v18, (__int64)v20);
      *(_BYTE *)(a1 + 16) |= 3u;
      v9 = v18 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_24;
    }
    if ( (_DWORD)v14 == 3 )
      break;
LABEL_18:
    if ( (v6 & 1) != 0 && v14 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  }
  v20[0] = (__int64)v21;
  v20[1] = 0x100000000LL;
  v16 = 0;
  v17 = 0;
  sub_A4B600(&v18, a2, HIDWORD(v14), v20, &v16);
  v3 = v19 & 1;
  v19 = (2 * v3) | v19 & 0xFD;
  if ( !(_BYTE)v3 )
  {
    if ( (_DWORD)v18 == 1 )
    {
      v3 = v16;
      v4 = v17;
      v12 = v16;
    }
    if ( (_BYTE *)v20[0] != v21 )
      _libc_free(v20[0], a2);
LABEL_17:
    v6 = v15;
    if ( (v15 & 2) != 0 )
      goto LABEL_26;
    goto LABEL_18;
  }
  sub_9C8CD0(&v13, &v18);
  v11 = v13;
  *(_BYTE *)(a1 + 16) |= 3u;
  *(_QWORD *)a1 = v11 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v19 & 2) != 0 )
    sub_9CE230(&v18);
  if ( (v19 & 1) != 0 && v18 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  if ( (_BYTE *)v20[0] != v21 )
    _libc_free(v20[0], &v18);
LABEL_25:
  v6 = v15;
  if ( (v15 & 2) != 0 )
LABEL_26:
    sub_9CEF10(&v14);
LABEL_8:
  if ( (v6 & 1) == 0 )
    return a1;
LABEL_9:
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  return a1;
}
