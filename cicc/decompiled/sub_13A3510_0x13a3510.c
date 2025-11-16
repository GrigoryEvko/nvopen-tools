// Function: sub_13A3510
// Address: 0x13a3510
//
__int64 __fastcall sub_13A3510(unsigned __int64 a1)
{
  unsigned int v1; // r12d
  _QWORD *v3; // rax
  __int64 v4; // rdx
  unsigned int v5; // r12d
  __int64 v6; // rbx
  __int64 v7; // r13

  if ( (a1 & 1) != 0 )
    return (unsigned int)sub_39FAC40(~(-1LL << (a1 >> 58)) & (a1 >> 1));
  v1 = (unsigned int)(*(_DWORD *)(a1 + 16) + 63) >> 6;
  if ( !v1 )
    return v1;
  v3 = *(_QWORD **)a1;
  v4 = v1 - 1;
  v5 = 0;
  v6 = *(_QWORD *)a1 + 8LL;
  v7 = v6 + 8 * v4;
  while ( 1 )
  {
    v5 += sub_39FAC40(*v3);
    v3 = (_QWORD *)v6;
    if ( v6 == v7 )
      break;
    v6 += 8;
  }
  return v5;
}
