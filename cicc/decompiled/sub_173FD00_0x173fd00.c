// Function: sub_173FD00
// Address: 0x173fd00
//
__int64 __fastcall sub_173FD00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(a2 + 32);
  if ( !v2 || a1 == v2 )
    return 0;
  else
    return v2 - 24;
}
