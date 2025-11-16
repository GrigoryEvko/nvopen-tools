// Function: sub_16041F0
// Address: 0x16041f0
//
_QWORD *__fastcall sub_16041F0(_QWORD *a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx

  v1 = 24LL * (*((_DWORD *)a1 + 5) & 0xFFFFFFF);
  result = &a1[v1 / 0xFFFFFFFFFFFFFFF8LL];
  if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
  {
    result = (_QWORD *)*(a1 - 1);
    a1 = &result[(unsigned __int64)v1 / 8];
  }
  for ( ; a1 != result; result += 3 )
  {
    if ( *result )
    {
      v3 = result[1];
      v4 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v4 = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = *(_QWORD *)(v3 + 16) & 3LL | v4;
    }
    *result = 0;
  }
  return result;
}
