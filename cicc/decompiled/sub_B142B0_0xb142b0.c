// Function: sub_B142B0
// Address: 0xb142b0
//
__int64 __fastcall sub_B142B0(__int64 a1, unsigned __int64 a2, char a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 result; // rax

  v4 = (__int64 *)(a1 + 8);
  if ( a3 )
    v4 = *(__int64 **)(a1 + 16);
  v5 = *v4;
  v6 = *(_QWORD *)a2;
  *(_QWORD *)(a2 + 8) = v4;
  v5 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)a2 = v5 | v6 & 7;
  *(_QWORD *)(v5 + 8) = a2;
  result = a2 | *v4 & 7;
  *v4 = result;
  *(_QWORD *)(a2 + 16) = a1;
  return result;
}
