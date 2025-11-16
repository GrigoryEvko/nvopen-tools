// Function: sub_68A290
// Address: 0x68a290
//
__int64 __fastcall sub_68A290(__int64 a1, _BYTE *a2, unsigned int a3)
{
  __int64 result; // rax
  int v5; // r8d

  if ( !(dword_4D048B4 | unk_4D0442C) )
    return 0;
  if ( !a3 || (v5 = sub_831280(a1, 0), result = a3, !v5) )
  {
    if ( a2 && (unsigned __int8)(a2[140] - 9) <= 2u )
      return *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a2 + 96LL) + 182LL) & 1;
    return 0;
  }
  return result;
}
