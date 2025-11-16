// Function: sub_1B91640
// Address: 0x1b91640
//
void __fastcall sub_1B91640(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi

  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a3 + 16) - 54) <= 1u )
      sub_1B1FC20(v3, a2, a3);
  }
}
