// Function: sub_14557D0
// Address: 0x14557d0
//
unsigned __int64 __fastcall sub_14557D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  __int64 v3; // r8
  __int64 v4; // rax
  __int64 v5; // r8
  unsigned __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 > 0x40 )
    return sub_16A5E70(a1, a2);
  v3 = (__int64)(*(_QWORD *)a1 << (64 - (unsigned __int8)v2)) >> (64 - (unsigned __int8)v2);
  v4 = v3 >> a2;
  v5 = v3 >> 63;
  if ( (_DWORD)a2 == v2 )
    v4 = v5;
  result = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & v4;
  *(_QWORD *)a1 = result;
  return result;
}
