// Function: sub_2DF5200
// Address: 0x2df5200
//
__int64 __fastcall sub_2DF5200(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  __int64 result; // rax
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 *v9; // r12

  v2 = *(_QWORD *)(a1 + 32);
  sub_2E31040(*(_QWORD *)(a1 + 16) + 40LL, v2);
  v3 = *a2;
  v4 = *(_QWORD *)v2;
  *(_QWORD *)(v2 + 8) = a2;
  v3 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v2 = v3 | v4 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  *a2 = *a2 & 7 | v2;
  while ( 1 )
  {
    v9 = (unsigned __int64 *)(**(_QWORD **)(a1 + 8) + 24LL);
    result = *(_QWORD *)(*(_QWORD *)a1 + 208LL) + 24LL * *(unsigned int *)(*(_QWORD *)a1 + 216LL);
    if ( v9 == (unsigned __int64 *)result )
      break;
    result = *(_QWORD *)(**(_QWORD **)(a1 + 8) + 32LL);
    if ( *(_QWORD *)(a1 + 24) != result )
      break;
    v6 = *v9;
    sub_2E31040(*(_QWORD *)(a1 + 16) + 40LL, *v9);
    v7 = *a2;
    v8 = *(_QWORD *)v6;
    *(_QWORD *)(v6 + 8) = a2;
    v7 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v6 = v7 | v8 & 7;
    *(_QWORD *)(v7 + 8) = v6;
    *a2 = *a2 & 7 | v6;
    **(_QWORD **)(a1 + 8) = v9;
  }
  return result;
}
