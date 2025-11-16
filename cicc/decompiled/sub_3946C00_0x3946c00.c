// Function: sub_3946C00
// Address: 0x3946c00
//
__int64 __fastcall sub_3946C00(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  int v4; // eax
  __int64 v5; // rax
  __int64 v7; // r14
  __int64 v8; // rbx

  v4 = sub_16D1B30((__int64 *)a1, a2, a3);
  if ( v4 != -1 )
  {
    v5 = *(_QWORD *)a1 + 8LL * v4;
    if ( v5 != *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) )
      return *(unsigned int *)(*(_QWORD *)v5 + 8LL);
  }
  if ( (unsigned __int8)sub_394A5C0(a1 + 32, a2, a3) )
    return 0;
  v7 = *(_QWORD *)(a1 + 128);
  if ( v7 == *(_QWORD *)(a1 + 120) )
    return 0;
  v8 = *(_QWORD *)(a1 + 120);
  while ( !(unsigned __int8)sub_16C9490(*(__int64 **)v8, (__int64)a2, a3, 0) )
  {
    v8 += 16;
    if ( v7 == v8 )
      return 0;
  }
  return *(unsigned int *)(v8 + 8);
}
