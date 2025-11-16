// Function: sub_2DF51A0
// Address: 0x2df51a0
//
unsigned __int64 *__fastcall sub_2DF51A0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // edx
  unsigned __int64 *result; // rax
  unsigned int v4; // esi

  v1 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v2 = *(_DWORD *)(v1 + 12) + 1;
  *(_DWORD *)(v1 + 12) = v2;
  result = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16));
  if ( v2 == *((_DWORD *)result - 2) )
  {
    result = *(unsigned __int64 **)a1;
    v4 = *(_DWORD *)(*(_QWORD *)a1 + 160LL);
    if ( v4 )
      return sub_F03D40((__int64 *)(a1 + 8), v4);
  }
  return result;
}
