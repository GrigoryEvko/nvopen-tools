// Function: sub_77ADB0
// Address: 0x77adb0
//
__int64 __fastcall sub_77ADB0(__int64 a1, __int64 a2, __int16 *a3, _DWORD *a4)
{
  __int64 v6; // rax
  int v7; // edx
  unsigned __int16 v8; // ax
  char *v9; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  int v13; // [rsp+8h] [rbp-38h] BYREF
  int v14; // [rsp+Ch] [rbp-34h] BYREF
  char *s; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v16[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( !qword_4D03C50 )
    goto LABEL_4;
  v6 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(char *)(qword_4D03C50 + 17LL) < 0 )
  {
    *(_BYTE *)(a1 + 136) |= 3u;
    if ( *(char *)(v6 + 11) >= 0 )
      return 0;
    v7 = 1;
  }
  else
  {
    v7 = 0;
    if ( *(char *)(v6 + 11) >= 0 )
      goto LABEL_4;
  }
  if ( *(_BYTE *)(v6 + 4) != 17 || (*(_BYTE *)(*(_QWORD *)(v6 + 216) + 206LL) & 2) == 0 )
  {
    *(_BYTE *)(a1 + 136) |= 2u;
    return 0;
  }
  if ( v7 )
    return 0;
LABEL_4:
  v8 = *(_WORD *)(a2 + 176);
  if ( v8 != 82 )
  {
    if ( v8 > 0x52u )
    {
      if ( v8 == 25771 )
      {
        v9 = sub_693F60((const char *)0x8C, 0);
        goto LABEL_9;
      }
      if ( v8 == 25772 )
      {
        sub_729E00(dword_4F07508[0], &s, v16, &v13, &v14);
        s = sub_722280(s);
        v9 = s;
        goto LABEL_9;
      }
    }
    else
    {
      switch ( v8 )
      {
        case 2u:
          sub_729E00(dword_4F07508[0], &s, v16, &v13, &v14);
          v9 = (char *)v16[0];
          if ( !v16[0] )
          {
            v9 = s;
            v16[0] = s;
          }
          goto LABEL_9;
        case 3u:
          v9 = sub_693F60((const char *)0x8A, 0);
LABEL_9:
          sub_77A8F0(a1, v9, (__int64)a3, a4);
          *((_BYTE *)a3 - 9) |= 1u;
          return 1;
        case 1u:
          v11 = sub_72BA30(6u);
          sub_7740C0(a1, (__int64)v11, a3, a4);
          return 1;
      }
    }
    sub_721090();
  }
  v12 = sub_72BA30(8u);
  sub_7744B0(a1, (__int64)v12, a3, a4);
  return 1;
}
