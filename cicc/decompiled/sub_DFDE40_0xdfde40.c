// Function: sub_DFDE40
// Address: 0xdfde40
//
__int64 __fastcall sub_DFDE40(
        _QWORD *a1,
        __int64 a2,
        _QWORD *a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 (__fastcall *v11)(__int64, __int64, _QWORD *, int, __int64, __int64, __int64, __int64, __int64); // rax
  int v12; // r14d
  unsigned int v13; // esi
  __int64 result; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r15
  int v18; // ebx

  v11 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD *, int, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 1432LL);
  if ( v11 != sub_DF6810 )
    return ((__int64 (__fastcall *)(_QWORD, __int64, _QWORD *))v11)(*a1, a2, a3);
  v12 = a9;
  if ( BYTE4(a9) )
  {
    v13 = 8 * a9;
  }
  else
  {
    v13 = 8;
    v12 = 1;
  }
  result = sub_BCD140(a3, v13);
  v17 = result;
  if ( a4 )
  {
    result = *(unsigned int *)(a2 + 8);
    v18 = 0;
    do
    {
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 8u, v15, v16);
        result = *(unsigned int *)(a2 + 8);
      }
      v18 += v12;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v17;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    while ( a4 != v18 );
  }
  return result;
}
