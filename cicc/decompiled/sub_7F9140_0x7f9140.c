// Function: sub_7F9140
// Address: 0x7f9140
//
__int64 __fastcall sub_7F9140(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 )
    return *(_QWORD *)(v1 + 8);
  else
    return *(_QWORD *)(a1 + 24);
}
