// Function: sub_74FC40
// Address: 0x74fc40
//
__int64 __fastcall sub_74FC40(__int64 a1, int a2, __int64 a3)
{
  bool v4; // zf
  unsigned __int64 v5; // rsi
  __int64 v7; // rax
  __int64 i; // rdi
  _DWORD v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(_BYTE *)(a3 + 136) == 0;
  v9[0] = a2;
  if ( !v4 && !dword_4F068C4 )
    return v9[0];
  sub_74A140(27, *(__int64 **)(a1 + 104), v9, (void (__fastcall **)(char *))a3);
  if ( *(char *)(a1 + 90) >= 0 || *(_BYTE *)(a3 + 136) && *(_BYTE *)(a3 + 141) )
    goto LABEL_4;
  if ( sub_736C60(6, *(__int64 **)(a1 + 104)) )
    sub_7450F0("__deprecated__", v9, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( !sub_736C60(21, *(__int64 **)(a1 + 104)) )
  {
LABEL_4:
    if ( (*(_BYTE *)(a1 + 144) & 1) != 0 )
      goto LABEL_21;
LABEL_5:
    v5 = *(unsigned int *)(a1 + 140);
    if ( (_DWORD)v5 )
      goto LABEL_19;
LABEL_6:
    if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(a1 + 120)) )
      return v9[0];
    goto LABEL_15;
  }
  sub_7450F0("__unavailable__", v9, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 144) & 1) == 0 )
    goto LABEL_5;
LABEL_21:
  sub_7450F0("__packed__", v9, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v5 = *(unsigned int *)(a1 + 140);
  if ( !(_DWORD)v5 )
    goto LABEL_6;
LABEL_19:
  sub_745D60("__aligned__", v5, v9, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(a1 + 120)) )
    return v9[0];
LABEL_15:
  v7 = sub_8D46C0(*(_QWORD *)(a1 + 120));
  if ( (unsigned int)sub_8D2310(v7) )
  {
    for ( i = sub_8D46C0(*(_QWORD *)(a1 + 120)); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    sub_745EC0(i, v9, a3);
  }
  return v9[0];
}
