// Function: sub_5C98E0
// Address: 0x5c98e0
//
__int64 __fastcall sub_5C98E0(__int64 a1, __int64 a2)
{
  __int64 i; // rax

  if ( unk_4D04964 )
    sub_684AA0(5, 2480, a1 + 56);
  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 181LL) |= 0x10u;
  return a2;
}
