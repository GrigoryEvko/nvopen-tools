// Function: sub_30D3E20
// Address: 0x30d3e20
//
__int64 __fastcall sub_30D3E20(__int64 a1, unsigned int a2, unsigned int a3, unsigned __int8 a4)
{
  __int64 v4; // r10
  int v5; // esi
  __int64 v6; // rax
  __int64 result; // rax

  v4 = a2;
  v5 = qword_5030168;
  if ( (_DWORD)v4 )
  {
    if ( !a4 )
    {
      sub_30D0F50(a1, 2 * (int)qword_5030168);
      v5 = qword_5030168;
    }
    return sub_30D0F50(a1, v5 * v4 + 2 * v5);
  }
  else if ( a3 <= 3 )
  {
    return sub_30D0F50(a1, 2 * (unsigned int)qword_5030168 * (a3 - a4));
  }
  else
  {
    v6 = 2 * (int)qword_5030168 * (3LL * (int)a3 / 2 - 1);
    if ( v6 >= 0x80000000LL )
      v6 = 0x7FFFFFFF;
    if ( v6 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      v6 = 0xFFFFFFFF80000000LL;
    result = *(int *)(a1 + 716) + v6;
    if ( result >= 0x80000000LL )
      result = 0x7FFFFFFF;
    if ( result <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      result = 0xFFFFFFFF80000000LL;
    *(_DWORD *)(a1 + 716) = result;
  }
  return result;
}
