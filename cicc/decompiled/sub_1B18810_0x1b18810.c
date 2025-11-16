// Function: sub_1B18810
// Address: 0x1b18810
//
__int64 __fastcall sub_1B18810(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r13
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rsi
  bool v9; // zf
  unsigned __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( sub_13F9E70(a2)
    && (v3 = sub_13FCB50(a2), v4 = sub_157EBA0(v3), v5 = v4, *(_BYTE *)(v4 + 16) == 26)
    && (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) == 3
    && (unsigned __int8)sub_1625AE0(v4, &v10, v11) )
  {
    v7 = v10;
    if ( v10 && (v8 = v11[0]) != 0 )
    {
      v9 = **(_QWORD **)(a2 + 32) == *(_QWORD *)(v5 - 24);
      *(_BYTE *)(a1 + 4) = 1;
      if ( v9 )
        *(_DWORD *)a1 = (v7 + (v8 >> 1)) / v8;
      else
        *(_DWORD *)a1 = (v8 + (v7 >> 1)) / v7;
    }
    else
    {
      *(_BYTE *)(a1 + 4) = 1;
      *(_DWORD *)a1 = 0;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 4) = 0;
  }
  return a1;
}
