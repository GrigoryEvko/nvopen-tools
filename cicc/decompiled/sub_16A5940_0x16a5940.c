// Function: sub_16A5940
// Address: 0x16a5940
//
__int64 __fastcall sub_16A5940(__int64 a1)
{
  _QWORD *v1; // rax
  unsigned int v2; // r12d
  __int64 v3; // rbx
  __int64 v4; // r13

  if ( !(((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6) )
    return 0;
  v1 = *(_QWORD **)a1;
  v2 = 0;
  v3 = *(_QWORD *)a1 + 8LL;
  v4 = v3 + 8LL * ((unsigned int)(((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6) - 1);
  while ( 1 )
  {
    v2 += sub_39FAC40(*v1);
    v1 = (_QWORD *)v3;
    if ( v3 == v4 )
      break;
    v3 += 8;
  }
  return v2;
}
