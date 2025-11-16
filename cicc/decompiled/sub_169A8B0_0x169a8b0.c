// Function: sub_169A8B0
// Address: 0x169a8b0
//
__int64 __fastcall sub_169A8B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // dl
  int v4; // ebx
  __int64 v5; // r14
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v10; // rcx
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(unsigned __int8 *)(a2 + 18);
  v3 = *(_BYTE *)(a2 + 18) & 7;
  if ( v3 == 1 )
  {
    v5 = *(_QWORD *)sub_16984A0(a2);
    v10 = *(_QWORD *)(sub_16984A0(a2) + 8);
    v2 = *(unsigned __int8 *)(a2 + 18);
    v8 = v10 & 0xFFFFFFFFFFFFLL;
    v7 = 0x7FFF000000000000LL;
  }
  else if ( v3 )
  {
    if ( v3 == 3 )
    {
      v8 = 0;
      v7 = 0;
      v5 = 0;
    }
    else
    {
      v4 = *(__int16 *)(a2 + 16) + 0x3FFF;
      v5 = *(_QWORD *)sub_16984A0(a2);
      v6 = *(_QWORD *)(sub_16984A0(a2) + 8);
      if ( v4 == 1 )
        v7 = v6 & 0x1000000000000LL;
      else
        v7 = ((__int64)v4 << 48) & 0x7FFF000000000000LL;
      v8 = v6 & 0xFFFFFFFFFFFFLL;
      v2 = *(unsigned __int8 *)(a2 + 18);
    }
  }
  else
  {
    v7 = 0x7FFF000000000000LL;
    v8 = 0;
    v5 = 0;
  }
  LOBYTE(v2) = (unsigned __int8)v2 >> 3;
  v11[0] = v5;
  v11[1] = v8 | v7 | (v2 << 63);
  sub_16A50F0(a1, 128, v11, 2);
  return a1;
}
