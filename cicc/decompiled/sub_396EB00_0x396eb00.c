// Function: sub_396EB00
// Address: 0x396eb00
//
__int64 __fastcall sub_396EB00(__int64 a1)
{
  __int64 v2; // r13

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 240) + 348LL) == 1
    && ((v2 = **(_QWORD **)(a1 + 264), (unsigned __int8)sub_1560180(v2 + 112, 56))
     || !(unsigned __int8)sub_1560180(v2 + 112, 30)
     || (*(_BYTE *)(v2 + 18) & 8) != 0) )
  {
    return 1;
  }
  else
  {
    return 2 * (unsigned int)(*(_BYTE *)(*(_QWORD *)(a1 + 272) + 1744LL) != 0);
  }
}
