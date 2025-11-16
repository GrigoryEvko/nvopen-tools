// Function: sub_737520
// Address: 0x737520
//
__int64 __fastcall sub_737520(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax

  v2 = *a1;
  if ( (*((_BYTE *)a1 + 25) & 3) == 0 )
    v2 = a1[1];
  sub_7258C0(a2, 6);
  result = *(unsigned __int8 *)(a2 + 168);
  *(_BYTE *)(a2 + 168) |= 1u;
  if ( (*((_BYTE *)a1 + 58) & 0x20) != 0 )
  {
    result = (unsigned int)result | 3;
    *(_BYTE *)(a2 + 168) = result;
  }
  *(_QWORD *)(a2 + 160) = v2;
  return result;
}
