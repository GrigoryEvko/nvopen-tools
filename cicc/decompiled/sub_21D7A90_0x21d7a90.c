// Function: sub_21D7A90
// Address: 0x21d7a90
//
__int64 __fastcall sub_21D7A90(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // r13
  unsigned int v5; // eax
  _DWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = 1;
  if ( (*(_BYTE *)(a2[1] + 792LL) & 2) != 0 )
    return v2;
  v4 = (_QWORD *)(*a2 + 112LL);
  LOBYTE(v5) = sub_15602E0(v4, "unsafe-fp-math", 0xEu);
  v2 = v5;
  if ( !(_BYTE)v5 )
    return v2;
  v8[0] = sub_1560340(v4, -1, "unsafe-fp-math", 0xEu);
  v6 = (_DWORD *)sub_155D8B0(v8);
  if ( v7 == 4 && *v6 == 1702195828 )
    return v2;
  else
    return 0;
}
