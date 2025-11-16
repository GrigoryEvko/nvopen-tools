// Function: sub_645420
// Address: 0x645420
//
__int64 sub_645420()
{
  __int64 v0; // rdi
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rcx

  v0 = unk_4F07470;
  sub_684AC0(unk_4F07470, 362);
  sub_7B8B50(v0, 362, v1, v2);
  if ( word_4F06418[0] != 1 || (sub_7BE840(0, 0) & 0xFFF7) != 0x43 )
    return 0;
  sub_7B8B50(0, 0, v4, v5);
  result = 1;
  if ( word_4F06418[0] == 67 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    sub_7B8B50(0, 0, v6, v7);
    do
      sub_7BE280(1, 40, 0, 0);
    while ( (unsigned int)sub_7BE800(67) );
    --*(_BYTE *)(qword_4F061C8 + 83LL);
    return 1;
  }
  return result;
}
