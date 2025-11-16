// Function: sub_169A7C0
// Address: 0x169a7c0
//
__int64 __fastcall sub_169A7C0(__int64 a1, __int64 a2)
{
  char v3; // al
  char v4; // dl
  int v5; // r13d
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 *v9; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = *(_BYTE *)(a2 + 18);
  v4 = v3 & 7;
  if ( (v3 & 7) == 1 )
  {
    v9 = (__int64 *)sub_16984A0(a2);
    v7 = 0x7FFF;
    v6 = *v9;
    v3 = *(_BYTE *)(a2 + 18);
  }
  else if ( !v4 || v4 == 3 )
  {
    v7 = 0;
    v6 = 0;
    if ( v4 != 3 )
    {
      v6 = 0x8000000000000000LL;
      v7 = 0x7FFF;
    }
  }
  else
  {
    v5 = *(__int16 *)(a2 + 16) + 0x3FFF;
    v6 = *(_QWORD *)sub_16984A0(a2);
    if ( v5 == 1 && v6 >= 0 )
    {
      v3 = *(_BYTE *)(a2 + 18);
      v7 = 0;
    }
    else
    {
      v3 = *(_BYTE *)(a2 + 18);
      v7 = v5 & 0x7FFF;
    }
  }
  v10[0] = v6;
  v10[1] = v7 | ((unsigned __int64)((v3 & 8) != 0) << 15);
  sub_16A50F0(a1, 80, v10, 2);
  return a1;
}
