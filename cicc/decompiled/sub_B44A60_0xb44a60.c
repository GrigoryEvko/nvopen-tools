// Function: sub_B44A60
// Address: 0xb44a60
//
__int64 __fastcall sub_B44A60(__int64 a1)
{
  __int64 v1; // rsi
  unsigned int *i; // rbx
  __int64 result; // rax

  v1 = 4;
  for ( i = (unsigned int *)&unk_3F2AA18; ; v1 = *i )
  {
    ++i;
    result = sub_B98000(a1, v1);
    if ( i == (unsigned int *)&unk_3F2AA24 )
      break;
  }
  return result;
}
