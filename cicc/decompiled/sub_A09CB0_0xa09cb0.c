// Function: sub_A09CB0
// Address: 0xa09cb0
//
__int64 *__fastcall sub_A09CB0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rcx
  const char **v4; // rsi
  int v5; // edx
  char v6; // al
  const char *v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // [rsp+8h] [rbp-288h]
  int v11; // [rsp+Ch] [rbp-284h]
  unsigned __int64 v12; // [rsp+18h] [rbp-278h] BYREF
  const char *v13; // [rsp+20h] [rbp-270h] BYREF
  char v14; // [rsp+28h] [rbp-268h]
  char v15; // [rsp+40h] [rbp-250h]
  char v16; // [rsp+41h] [rbp-24Fh]
  unsigned __int64 v17; // [rsp+50h] [rbp-240h] BYREF
  __int64 v18; // [rsp+58h] [rbp-238h]
  _BYTE v19[560]; // [rsp+60h] [rbp-230h] BYREF

  sub_A4DCE0(&v17, *(_QWORD *)(a2 + 240), 22, 0);
  if ( (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v17 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v18 = 0x4000000000LL;
    v17 = (unsigned __int64)v19;
    while ( 1 )
    {
      do
      {
        v4 = *(const char ***)(a2 + 240);
        sub_9CEFB0((__int64)&v13, (__int64)v4, 0, v3);
        if ( (v14 & 1) != 0 )
        {
          v14 &= ~2u;
          v8 = v13;
          v13 = 0;
          v12 = (unsigned __int64)v8 | 1;
          v9 = (unsigned __int64)v8 & 0xFFFFFFFFFFFFFFFELL;
          if ( v9 )
            goto LABEL_15;
        }
        else
        {
          v12 = 1;
          v10 = HIDWORD(v13);
          v11 = (int)v13;
        }
        if ( v11 == 1 )
        {
          *a1 = 1;
          goto LABEL_16;
        }
        if ( (v11 & 0xFFFFFFFD) == 0 )
        {
          v4 = &v13;
          v16 = 1;
          v13 = "Malformed block";
          v15 = 3;
          sub_A01DB0(a1, (__int64)&v13);
          goto LABEL_16;
        }
        v4 = *(const char ***)(a2 + 240);
        LODWORD(v18) = 0;
        sub_A4B600(&v13, v4, v10, &v17, 0);
        v5 = v14 & 1;
        v3 = (unsigned int)(2 * v5);
        v6 = (2 * v5) | v14 & 0xFD;
        v14 = v6;
        if ( (_BYTE)v5 )
        {
          v14 = v6 & 0xFD;
          v9 = (unsigned __int64)v13;
          v13 = 0;
LABEL_15:
          *a1 = v9 | 1;
          goto LABEL_16;
        }
      }
      while ( (_DWORD)v13 != 6 );
      v4 = (const char **)a2;
      sub_A09940((__int64 *)&v12, a2, (__int64 **)&v17);
      if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( (v14 & 2) != 0 )
        goto LABEL_11;
      if ( (v14 & 1) != 0 && v13 )
        (*(void (__fastcall **)(const char *))(*(_QWORD *)v13 + 8LL))(v13);
    }
    *a1 = v12 & 0xFFFFFFFFFFFFFFFELL | 1;
    if ( (v14 & 2) != 0 )
LABEL_11:
      sub_9CE230(&v13);
    if ( (v14 & 1) != 0 && v13 )
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v13 + 8LL))(v13);
LABEL_16:
    if ( (_BYTE *)v17 != v19 )
      _libc_free(v17, v4);
  }
  return a1;
}
