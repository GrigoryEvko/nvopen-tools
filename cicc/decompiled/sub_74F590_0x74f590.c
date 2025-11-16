// Function: sub_74F590
// Address: 0x74f590
//
__int64 __fastcall sub_74F590(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // r12
  bool v5; // zf
  char i; // al
  char v8; // al
  __int64 v9; // rax
  __int64 j; // rdi
  _DWORD v11[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = a1;
  v5 = *(_BYTE *)(a3 + 136) == 0;
  v11[0] = a2;
  if ( !v5 && !dword_4F068C4 )
    return v11[0];
  if ( *(_BYTE *)(a1 + 140) == 7 )
  {
LABEL_4:
    if ( (*(_BYTE *)(a1 + 143) & 1) == 0 )
      goto LABEL_5;
LABEL_17:
    sub_7450F0("__unused__", v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    if ( *(char *)(a1 + 90) >= 0 )
      goto LABEL_6;
LABEL_18:
    if ( !*(_BYTE *)(a3 + 136) || !*(_BYTE *)(a3 + 141) )
    {
      if ( sub_736C60(6, *(__int64 **)(a1 + 104)) )
        sub_7450F0("__deprecated__", v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
      if ( sub_736C60(21, *(__int64 **)(a1 + 104)) )
        sub_7450F0("__unavailable__", v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    }
    goto LABEL_6;
  }
  sub_745E10(a1, v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 143) & 2) != 0 )
    sub_7450F0("__may_alias__", v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v8 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v8 - 9) > 2u )
  {
    if ( v8 == 2 && (*(_BYTE *)(a1 + 161) & 8) != 0 && (!*(_BYTE *)(a3 + 136) || !*(_BYTE *)(a3 + 141)) )
      sub_745150(*(_BYTE *)(a1 + 163) & 7, v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    goto LABEL_4;
  }
  if ( *(_BYTE *)(a3 + 136) && *(_BYTE *)(a3 + 141) )
    goto LABEL_4;
  sub_745150(*(_BYTE *)(*(_QWORD *)(a1 + 168) + 109LL) & 7, v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 143) & 1) != 0 )
    goto LABEL_17;
LABEL_5:
  if ( *(char *)(a1 + 90) < 0 )
    goto LABEL_18;
LABEL_6:
  sub_74A140(27, *(__int64 **)(a1 + 104), v11, (void (__fastcall **)(char *))a3);
  sub_74A140(51, *(__int64 **)(a1 + 104), v11, (void (__fastcall **)(char *))a3);
  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(v4 + 140) )
    v4 = *(_QWORD *)(v4 + 160);
  if ( i == 11 && (*(_BYTE *)(v4 + 179) & 0x10) != 0 )
    sub_7450F0("__transparent_union__", v11, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (unsigned int)sub_8D2E30(v4) )
  {
    v9 = sub_8D46C0(v4);
    if ( (unsigned int)sub_8D2310(v9) )
    {
      for ( j = sub_8D46C0(v4); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      sub_745EC0(j, v11, a3);
    }
  }
  return v11[0];
}
