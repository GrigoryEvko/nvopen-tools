// Function: sub_C6EB10
// Address: 0xc6eb10
//
__int64 __fastcall sub_C6EB10(__int64 a1, char *a2, __int64 a3)
{
  const char *v3; // rsi
  __int64 v4; // rax
  unsigned __int8 *v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  __int64 v9; // [rsp+8h] [rbp-88h] BYREF
  __int64 v10; // [rsp+10h] [rbp-80h] BYREF
  char v11; // [rsp+18h] [rbp-78h]
  char *v12; // [rsp+20h] [rbp-70h]
  char *v13; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v14; // [rsp+30h] [rbp-60h]
  unsigned __int16 v15[40]; // [rsp+40h] [rbp-50h] BYREF

  v14 = (unsigned __int8 *)&a2[a3];
  v11 = 0;
  v12 = a2;
  v13 = a2;
  v15[0] = 0;
  if ( !(unsigned __int8)sub_C6A630(a2, a3, &v9) )
  {
    v3 = "Invalid UTF-8 sequence";
    v13 = &v12[v9];
    if ( !(unsigned __int8)sub_C68D40(&v10, (__int64)"Invalid UTF-8 sequence") )
      goto LABEL_3;
  }
  v3 = (const char *)v15;
  if ( !(unsigned __int8)sub_C6DCD0(&v10, v15) )
    goto LABEL_3;
  v6 = (unsigned __int8 *)v13;
  v7 = 0x100002600LL;
  if ( v13 == (char *)v14 )
    goto LABEL_14;
  while ( 1 )
  {
    v8 = *v6;
    if ( (unsigned __int8)v8 > 0x20u || !_bittest64(&v7, v8) )
      break;
    v13 = (char *)++v6;
    if ( v6 == v14 )
      goto LABEL_14;
  }
  v3 = "Text after end of document";
  if ( (unsigned __int8)sub_C68D40(&v10, (__int64)"Text after end of document") )
  {
LABEL_14:
    v3 = (const char *)v15;
    *(_BYTE *)(a1 + 40) = *(_BYTE *)(a1 + 40) & 0xFC | 2;
    sub_C6A4F0(a1, v15);
  }
  else
  {
LABEL_3:
    v4 = v10;
    *(_BYTE *)(a1 + 40) |= 3u;
    v10 = 0;
    *(_QWORD *)a1 = v4 & 0xFFFFFFFFFFFFFFFELL;
  }
  sub_C6BC50(v15);
  if ( v11 )
  {
    v11 = 0;
    if ( (v10 & 1) != 0 || (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v10, (__int64)v3);
  }
  return a1;
}
