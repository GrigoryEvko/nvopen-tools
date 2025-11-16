// Function: sub_9C3450
// Address: 0x9c3450
//
__int64 __fastcall sub_9C3450(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rbx
  unsigned int v4; // eax
  int v5; // eax

  v1 = a1[69];
  v2 = a1[68];
  v3 = a1[53];
  v4 = sub_C92610(v2, v1);
  v5 = sub_C92860(v3 + 48, v2, v1, v4);
  if ( v5 == -1 )
    return *(_QWORD *)(*(_QWORD *)(v3 + 48) + 8LL * *(unsigned int *)(v3 + 56));
  else
    return *(_QWORD *)(*(_QWORD *)(v3 + 48) + 8LL * v5);
}
