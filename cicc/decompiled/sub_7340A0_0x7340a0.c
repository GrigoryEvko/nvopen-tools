// Function: sub_7340A0
// Address: 0x7340a0
//
__int64 __fastcall sub_7340A0(__int64 a1)
{
  __int64 i; // r8

  for ( i = a1; (unsigned __int8)(*(_BYTE *)i - 1) > 1u; i = *(_QWORD *)(i + 32) )
    ;
  return i;
}
