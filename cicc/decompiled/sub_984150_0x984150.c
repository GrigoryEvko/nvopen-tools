// Function: sub_984150
// Address: 0x984150
//
bool __fastcall sub_984150(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax

  v3 = a2;
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(a2 + 16);
  v4 = sub_BCAC60(v3);
  return (sub_B2DB90(a1, v4) & 0xFD) == 0;
}
