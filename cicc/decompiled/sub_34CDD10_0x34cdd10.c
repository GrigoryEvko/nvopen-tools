// Function: sub_34CDD10
// Address: 0x34cdd10
//
__int64 __fastcall sub_34CDD10(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  int v11; // r14d
  unsigned int v12; // esi
  __int64 result; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  int v17; // r12d

  v11 = a9;
  if ( BYTE4(a9) )
  {
    v12 = 8 * a9;
  }
  else
  {
    v12 = 8;
    v11 = 1;
  }
  result = sub_BCD140(a3, v12);
  v16 = result;
  if ( a4 )
  {
    result = *(unsigned int *)(a2 + 8);
    v17 = 0;
    do
    {
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 8u, v14, v15);
        result = *(unsigned int *)(a2 + 8);
      }
      v17 += v11;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v16;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    while ( a4 != v17 );
  }
  return result;
}
