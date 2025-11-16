// Function: sub_74F860
// Address: 0x74f860
//
__int64 __fastcall sub_74F860(__int64 a1, int a2, __int64 a3)
{
  bool v5; // zf
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rsi
  char v8; // al
  char *v9; // rdi
  char v10; // al
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 i; // rdi
  _DWORD v15[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_BYTE *)(a3 + 136) == 0;
  v15[0] = a2;
  if ( !v5 && !dword_4F068C4 )
    return v15[0];
  sub_74A140(27, *(__int64 **)(a1 + 104), v15, (void (__fastcall **)(char *))a3);
  v6 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v6 )
  {
    sub_745D60("__aligned__", v6, v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    v7 = *(unsigned __int16 *)(a1 + 158);
    if ( !(_WORD)v7 )
      goto LABEL_7;
  }
  else
  {
    v7 = *(unsigned __int16 *)(a1 + 158);
    if ( !(_WORD)v7 )
      goto LABEL_7;
  }
  if ( !*(_BYTE *)(a3 + 136) || !*(_BYTE *)(a3 + 141) )
  {
    sub_745D60("__init_priority__", v7, v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    if ( !*(_QWORD *)(a1 + 160) )
      goto LABEL_11;
    goto LABEL_8;
  }
LABEL_7:
  if ( !*(_QWORD *)(a1 + 160) )
    goto LABEL_11;
LABEL_8:
  if ( v15[0] )
    (*(void (__fastcall **)(char *, __int64))a3)(" ", a3);
  v15[0] = 1;
  (*(void (__fastcall **)(const char *, __int64))a3)("__attribute__((cleanup(", a3);
  sub_74C010(*(_QWORD *)(a1 + 160), 11, a3);
  (*(void (__fastcall **)(const char *, __int64))a3)(")))", a3);
LABEL_11:
  sub_74A140(39, *(__int64 **)(a1 + 104), v15, (void (__fastcall **)(char *))a3);
  sub_745150(*(_BYTE *)(a1 + 168) & 7, v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 168) & 0x18) == 8 )
    sub_7450F0("__weak__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 91) & 4) != 0 )
  {
    sub_7450F0("__unused__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    if ( (*(_BYTE *)(a1 + 168) & 0x40) == 0 )
    {
LABEL_15:
      if ( *(char *)(a1 + 90) >= 0 )
        goto LABEL_16;
      goto LABEL_30;
    }
  }
  else if ( (*(_BYTE *)(a1 + 168) & 0x40) == 0 )
  {
    goto LABEL_15;
  }
  sub_7450F0("__used__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( *(char *)(a1 + 90) >= 0 )
    goto LABEL_16;
LABEL_30:
  if ( *(_BYTE *)(a3 + 136) && *(_BYTE *)(a3 + 141) )
    goto LABEL_16;
  if ( sub_736C60(6, *(__int64 **)(a1 + 104)) )
    sub_7450F0("__deprecated__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( !sub_736C60(21, *(__int64 **)(a1 + 104)) )
  {
LABEL_16:
    v8 = *(_BYTE *)(a1 + 169);
    if ( (v8 & 1) == 0 )
      goto LABEL_17;
LABEL_42:
    sub_7450F0("__nocommon__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    if ( *(char *)(a1 + 169) >= 0 )
      goto LABEL_18;
    goto LABEL_27;
  }
  sub_7450F0("__unavailable__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v8 = *(_BYTE *)(a1 + 169);
  if ( (v8 & 1) != 0 )
    goto LABEL_42;
LABEL_17:
  if ( v8 >= 0 )
    goto LABEL_18;
LABEL_27:
  v13 = *(_QWORD *)(a1 + 128);
  if ( v13 && (*(_BYTE *)(v13 + 34) & 4) != 0 )
    sub_7450F0("__transparent_union__", v15, (__int64 (__fastcall **)(const char *, _QWORD))a3);
LABEL_18:
  v9 = *(char **)(a1 + 224);
  if ( v9 )
    sub_747170(v9, v15, a3);
  if ( (*(_BYTE *)(a1 + 140) & 1) != 0 )
  {
    sub_74A140(62, *(__int64 **)(a1 + 104), v15, (void (__fastcall **)(char *))a3);
    v10 = *(_BYTE *)(a1 + 168);
    if ( (v10 & 0x20) == 0 )
    {
LABEL_22:
      if ( (v10 & 0x10) != 0 )
        sub_74A140(79, *(__int64 **)(a1 + 104), v15, (void (__fastcall **)(char *))a3);
      goto LABEL_24;
    }
  }
  else
  {
    v10 = *(_BYTE *)(a1 + 168);
    if ( (v10 & 0x20) == 0 )
      goto LABEL_22;
  }
  sub_74A140(26, *(__int64 **)(a1 + 104), v15, (void (__fastcall **)(char *))a3);
LABEL_24:
  if ( (unsigned int)sub_8D2E30(*(_QWORD *)(a1 + 120)) )
  {
    v11 = sub_8D46C0(*(_QWORD *)(a1 + 120));
    if ( (unsigned int)sub_8D2310(v11) )
    {
      for ( i = sub_8D46C0(*(_QWORD *)(a1 + 120)); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      sub_745EC0(i, v15, a3);
    }
  }
  return v15[0];
}
