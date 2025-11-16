// Function: sub_7741A0
// Address: 0x7741a0
//
__int64 __fastcall sub_7741A0(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 result; // rax

  v7 = sub_773040(*a4);
  if ( v7 )
  {
    *(_BYTE *)a5 = 59;
    *(_DWORD *)(a5 + 16) = 0;
    *(_QWORD *)(a5 + 8) = v7;
    return 1;
  }
  else
  {
    result = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xD2Du, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
  }
  return result;
}
