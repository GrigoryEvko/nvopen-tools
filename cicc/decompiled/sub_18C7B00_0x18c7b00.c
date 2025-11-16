// Function: sub_18C7B00
// Address: 0x18c7b00
//
__int64 __fastcall sub_18C7B00(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v3; // rdi

  if ( qword_4FADF60 == qword_4FADF68 )
    return 0;
  for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    v3 = i - 56;
    if ( !i )
      v3 = 0;
    sub_18C6ED0(v3);
  }
  return 1;
}
