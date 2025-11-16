// Function: sub_7A6D50
// Address: 0x7a6d50
//
__int64 __fastcall sub_7A6D50(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rdx

  result = unk_4F06AC0;
  if ( unk_4F06AC0 && (v2 = *(_QWORD *)(a1 + 8), result = unk_4F06AC0 - 1LL, v2 <= unk_4F06AC0 - 1LL) )
  {
    *(_QWORD *)(a1 + 8) = v2 + 1;
    *(_QWORD *)(a1 + 16) = 0;
  }
  else
  {
    if ( !*(_BYTE *)(a1 + 28) )
    {
      result = sub_6851C0((unsigned int)(dword_4F077C4 != 2) + 103, dword_4F07508);
      *(_BYTE *)(a1 + 28) = 1;
    }
    *(_QWORD *)(a1 + 16) = 0;
  }
  return result;
}
