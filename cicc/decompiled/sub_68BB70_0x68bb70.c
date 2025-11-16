// Function: sub_68BB70
// Address: 0x68bb70
//
__int64 __fastcall sub_68BB70(_QWORD *a1, _QWORD *a2, _DWORD *a3, __int64 a4, _DWORD *a5)
{
  __int64 result; // rax

  result = unk_4D03C50;
  if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_7306C0(*a1) || (result = sub_7306C0(*a2), (_DWORD)result) )
    {
      if ( (unsigned int)sub_6E5430() )
        sub_6851C0(0x369u, a3);
      sub_6E6260(a4);
      sub_6E6450(a1);
      result = sub_6E6450(a2);
      *a5 = 1;
    }
  }
  return result;
}
