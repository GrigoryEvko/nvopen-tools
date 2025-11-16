// Function: sub_16055B0
// Address: 0x16055b0
//
__int64 __fastcall sub_16055B0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax

  v3 = sub_16D1B30(a1 + 2864, a2, a3);
  if ( v3 == -1 )
    return *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 2864) + 8LL * *(unsigned int *)(a1 + 2872)) + 8LL);
  else
    return *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 2864) + 8LL * v3) + 8LL);
}
