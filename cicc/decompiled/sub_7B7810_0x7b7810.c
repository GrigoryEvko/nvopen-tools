// Function: sub_7B7810
// Address: 0x7b7810
//
__int64 sub_7B7810()
{
  char *v0; // rax
  char v1; // cl
  __int64 v3; // [rsp+8h] [rbp-18h] BYREF

  v3 = 0;
  v0 = qword_4F06460++;
  v1 = *v0;
  if ( *v0 == 60 )
    v1 = 62;
  if ( (unsigned int)sub_7B6B00(&v3, 1, 17, v1, 0, -1, v0, 0) )
  {
    unk_4F06208 = 8;
    if ( unk_4D03D20 )
    {
      return 0;
    }
    else
    {
      sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
      sub_684AC0(8u, 8u);
      return 11;
    }
  }
  else
  {
    ++qword_4F06460;
    return 11;
  }
}
