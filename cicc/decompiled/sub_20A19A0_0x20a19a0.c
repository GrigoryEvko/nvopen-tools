// Function: sub_20A19A0
// Address: 0x20a19a0
//
__int64 __fastcall sub_20A19A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // eax

  result = sub_17004D0(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a2 + 88) + 40LL), *(_BYTE **)(a2 + 88));
  if ( (_BYTE)result )
  {
    LOBYTE(v3) = sub_20A1950(a1);
    return v3 ^ 1u;
  }
  return result;
}
