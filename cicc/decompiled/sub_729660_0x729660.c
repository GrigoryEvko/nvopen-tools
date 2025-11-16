// Function: sub_729660
// Address: 0x729660
//
__int64 __fastcall sub_729660(char a1)
{
  __int64 result; // rax
  unsigned __int64 v3; // rdi

  result = unk_4F06C40;
  v3 = unk_4F06C40 + 1LL;
  if ( qword_4F06C48 < (unsigned __int64)(unk_4F06C40 + 1LL) )
  {
    sub_729510(v3);
    result = unk_4F06C40;
    v3 = unk_4F06C40 + 1LL;
  }
  unk_4F06C40 = v3;
  *((_BYTE *)qword_4F06C50 + result) = a1;
  return result;
}
