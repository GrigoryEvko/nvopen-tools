// Function: sub_27792F0
// Address: 0x27792f0
//
__int64 __fastcall sub_27792F0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d

  LOBYTE(v2) = a2 == -4096 || a1 == -4096 || a1 == -8192 || a2 == -8192;
  if ( (_BYTE)v2 )
  {
    LOBYTE(v2) = a2 == a1;
    return v2;
  }
  if ( ((unsigned __int8)sub_A73ED0((_QWORD *)(a1 + 72), 6) || (unsigned __int8)sub_B49560(a1, 6))
    && *(_QWORD *)(a2 + 40) != *(_QWORD *)(a1 + 40) )
  {
    return v2;
  }
  return sub_B46130(a1, a2, 1);
}
