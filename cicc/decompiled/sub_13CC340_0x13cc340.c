// Function: sub_13CC340
// Address: 0x13cc340
//
__int64 __fastcall sub_13CC340(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rsi

  if ( *(_DWORD *)(a1 + 8) > 0x40u || *((_DWORD *)a2 + 2) > 0x40u )
    return sub_16A51C0(a1, a2);
  v3 = *a2;
  *(_QWORD *)a1 = *a2;
  result = *((unsigned int *)a2 + 2);
  *(_DWORD *)(a1 + 8) = result;
  v4 = 0xFFFFFFFFFFFFFFFFLL >> -(char)result;
  if ( (unsigned int)result > 0x40 )
  {
    result = (unsigned int)((unsigned __int64)(result + 63) >> 6) - 1;
    *(_QWORD *)(v3 + 8 * result) &= v4;
  }
  else
  {
    *(_QWORD *)a1 = v4 & v3;
  }
  return result;
}
