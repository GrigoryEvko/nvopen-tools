// Function: sub_72F9F0
// Address: 0x72f9f0
//
__int64 __fastcall sub_72F9F0(__int64 a1, __int64 a2, _BYTE *a3, _QWORD *a4)
{
  __int64 result; // rax
  _QWORD *v7; // rdi

  result = *(unsigned __int8 *)(a1 + 177);
  if ( (_BYTE)result == 4 )
  {
    if ( !a2 )
    {
      if ( (*(_BYTE *)(a1 + 89) & 2) != 0 )
        a2 = sub_72F070(a1);
      else
        a2 = *(_QWORD *)(a1 + 40);
    }
    v7 = sub_72F9B0(a1, a2);
    result = *((unsigned __int8 *)v7 + 16);
    *a3 = result;
    *a4 = v7 + 3;
  }
  else
  {
    *a3 = result;
    *a4 = a1 + 184;
  }
  return result;
}
