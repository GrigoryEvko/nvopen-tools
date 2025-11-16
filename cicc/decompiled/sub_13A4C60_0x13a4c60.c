// Function: sub_13A4C60
// Address: 0x13a4c60
//
__int64 __fastcall sub_13A4C60(__int64 a1, unsigned __int8 a2)
{
  int v2; // ecx
  unsigned __int64 v3; // rdx
  unsigned int v4; // r12d
  __int64 result; // rax
  int v6; // ecx
  _QWORD *v7; // rdx

  v2 = *(_DWORD *)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = (unsigned int)(v2 + 63) >> 6;
  result = v4;
  if ( v4 < v3 )
  {
    result = (__int64)memset((void *)(*(_QWORD *)a1 + 8LL * v4), -a2, 8 * (v3 - v4));
    v2 = *(_DWORD *)(a1 + 16);
  }
  v6 = v2 & 0x3F;
  if ( v6 )
  {
    result = -1LL << v6;
    v7 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (v4 - 1));
    if ( a2 )
    {
      *v7 |= result;
    }
    else
    {
      result = ~result;
      *v7 &= result;
    }
  }
  return result;
}
