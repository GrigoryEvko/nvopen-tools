// Function: sub_14E9E10
// Address: 0x14e9e10
//
__int64 __fastcall sub_14E9E10(_QWORD *a1)
{
  __int64 v1; // rbx
  int v2; // eax

  v1 = a1[53];
  v2 = sub_16D1B30(v1 + 48, a1[68], a1[69]);
  if ( v2 == -1 )
    return *(_QWORD *)(*(_QWORD *)(v1 + 48) + 8LL * *(unsigned int *)(v1 + 56));
  else
    return *(_QWORD *)(*(_QWORD *)(v1 + 48) + 8LL * v2);
}
