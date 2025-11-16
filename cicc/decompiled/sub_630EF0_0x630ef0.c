// Function: sub_630EF0
// Address: 0x630ef0
//
__int64 __fastcall sub_630EF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r13

  result = 0;
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || (result = dword_4F07774) != 0) )
  {
    result = 0;
    if ( *(_BYTE *)(a1 + 8) == 1 )
    {
      v3 = *(_QWORD *)(a1 + 24);
      if ( v3 )
      {
        if ( !*(_QWORD *)v3 && !*(_BYTE *)(v3 + 8) )
        {
          v4 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 8LL);
          return (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
              && (unsigned int)sub_8DC1A0(v4)
              || (unsigned int)sub_8DF8D0(a2, v4) != 0;
        }
      }
    }
  }
  return result;
}
