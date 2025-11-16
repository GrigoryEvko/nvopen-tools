// Function: sub_1210F40
// Address: 0x1210f40
//
__int64 __fastcall sub_1210F40(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v2; // rsi
  unsigned int v4; // r12d
  bool v5; // al
  const char *v6; // [rsp+10h] [rbp-40h] BYREF
  char v7; // [rsp+30h] [rbp-20h]
  char v8; // [rsp+31h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 240) == 529 && *(_BYTE *)(a1 + 332) )
  {
    v4 = *(_DWORD *)(a1 + 328);
    if ( v4 <= 0x40 )
      v5 = *(_QWORD *)(a1 + 320) == 0;
    else
      v5 = v4 == (unsigned int)sub_C444A0(a1 + 320);
    *a2 = !v5;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 232);
    v8 = 1;
    v6 = "expected integer";
    v7 = 3;
    sub_11FD800(a1 + 176, v2, (__int64)&v6, 1);
    return 1;
  }
}
