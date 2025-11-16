// Function: sub_652340
// Address: 0x652340
//
__int64 __fastcall sub_652340(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = a1[44];
  if ( result && *(_QWORD *)result && (unsigned __int8)(*(_BYTE *)(*a1 + 80LL) - 10) <= 1u )
  {
    v2 = *(_QWORD *)(*a1 + 88LL);
    if ( *(_BYTE *)(result + 16) == 11 )
      *(_BYTE *)(v2 + 206) |= 0x40u;
    else
      *(_BYTE *)(*(_QWORD *)(result + 24) + 57LL) |= 2u;
    return sub_869D70(v2, 11);
  }
  return result;
}
