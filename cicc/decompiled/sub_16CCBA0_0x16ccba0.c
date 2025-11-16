// Function: sub_16CCBA0
// Address: 0x16ccba0
//
_QWORD *__fastcall sub_16CCBA0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  int v4; // ecx
  _QWORD *result; // rax
  bool v6; // cf

  v3 = *(_DWORD *)(a1 + 24);
  v4 = 2 * v3;
  if ( 4 * (*(_DWORD *)(a1 + 28) - *(_DWORD *)(a1 + 32)) >= 3 * v3 )
  {
    v6 = v3 < 0x40;
    v3 = 128;
    if ( !v6 )
      v3 = v4;
  }
  else if ( v3 - *(_DWORD *)(a1 + 28) >= v3 >> 3 )
  {
    goto LABEL_4;
  }
  sub_16CCA80(a1, v3);
LABEL_4:
  result = sub_16CC9F0(a1, a2);
  if ( *result != a2 )
  {
    if ( *result == -2 )
      --*(_DWORD *)(a1 + 32);
    else
      ++*(_DWORD *)(a1 + 28);
    *result = a2;
    ++*(_QWORD *)a1;
  }
  return result;
}
