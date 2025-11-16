// Function: sub_2AAACB0
// Address: 0x2aaacb0
//
__int64 __fastcall sub_2AAACB0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rbx
  bool v4; // r13
  __int64 result; // rax
  __int64 v6; // r12

  v2 = *(_DWORD *)a2;
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
  if ( *(_BYTE *)(a2 + 4) )
  {
    v4 = v2 != 0;
    if ( *(_DWORD *)(v3 + 96) )
      return 1;
  }
  else
  {
    v4 = v2 > 1;
    if ( *(_DWORD *)(v3 + 96) )
      return 1;
  }
  v6 = sub_D46F00(*(_QWORD *)(v3 + 416));
  if ( v6 == sub_D47930(*(_QWORD *)(v3 + 416))
    || (result = (unsigned __int8)byte_500CDA8, byte_500CDA8)
    && (result = *(unsigned __int8 *)(*(_QWORD *)(v3 + 440) + 664LL), (_BYTE)result) )
  {
    if ( v4 )
      return *(unsigned __int8 *)(*(_QWORD *)(v3 + 504) + 40LL) ^ 1u;
    return 1;
  }
  return result;
}
