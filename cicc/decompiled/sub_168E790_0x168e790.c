// Function: sub_168E790
// Address: 0x168e790
//
__int64 __fastcall sub_168E790(__int64 a1, _BYTE *a2)
{
  __int64 *v2; // rsi
  __int64 v3; // rdx
  _QWORD *v4; // rsi
  int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rax

  if ( (*a2 & 4) != 0 )
  {
    v2 = (__int64 *)*((_QWORD *)a2 - 1);
    v3 = *v2;
    v4 = v2 + 2;
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  v5 = sub_16D1B30(a1 + 272, v4, v3);
  if ( v5 == -1 )
    return 0;
  v6 = *(_QWORD *)(a1 + 272);
  v7 = v6 + 8LL * v5;
  if ( v7 == v6 + 8LL * *(unsigned int *)(a1 + 280) )
    return 0;
  else
    return *(unsigned int *)(*(_QWORD *)v7 + 8LL);
}
