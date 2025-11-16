// Function: sub_372F890
// Address: 0x372f890
//
__int64 __fastcall sub_372F890(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 result; // rax

  v2 = *a1;
  v3 = *(char *)(*a1 + 48);
  if ( (_BYTE)v3 == 2 )
  {
    *(_DWORD *)v2 = *(_DWORD *)a2;
    result = *((unsigned __int16 *)a2 + 2);
    *(_WORD *)(v2 + 4) = result;
  }
  else
  {
    if ( (_BYTE)v3 != 0xFF )
      funcs_32198D3[v3]();
    *(_BYTE *)(v2 + 48) = 2;
    result = *a2;
    *(_QWORD *)v2 = *a2;
  }
  return result;
}
