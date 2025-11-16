// Function: sub_2E88E20
// Address: 0x2e88e20
//
__int64 __fastcall sub_2E88E20(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // r14
  _QWORD *v3; // r13
  __int64 v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 *v6; // rcx
  unsigned __int64 v7; // rdx

  result = a1;
  v2 = (_QWORD *)a1;
  if ( (*(_BYTE *)a1 & 4) == 0 && (*(_BYTE *)(a1 + 44) & 8) != 0 )
  {
    do
      result = *(_QWORD *)(result + 8);
    while ( (*(_BYTE *)(result + 44) & 8) != 0 );
  }
  v3 = *(_QWORD **)(result + 8);
  v4 = *(_QWORD *)(a1 + 24) + 40LL;
  if ( (_QWORD *)a1 != v3 )
  {
    do
    {
      v5 = v2;
      v2 = (_QWORD *)v2[1];
      sub_2E31080(v4, (__int64)v5);
      v6 = (unsigned __int64 *)v5[1];
      v7 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
      *v6 = v7 | *v6 & 7;
      *(_QWORD *)(v7 + 8) = v6;
      *v5 &= 7uLL;
      v5[1] = 0;
      result = sub_2E310F0(v4);
    }
    while ( v2 != v3 );
  }
  return result;
}
