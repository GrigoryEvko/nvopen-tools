// Function: sub_BCBBB0
// Address: 0xbcbbb0
//
__int64 __fastcall sub_BCBBB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v5; // eax
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rax

  v4 = *a1;
  v5 = sub_C92610(a2, a3);
  v6 = sub_C92860(v4 + 2968, a2, a3, v5);
  if ( v6 == -1 )
    return 0;
  v7 = *(_QWORD *)(v4 + 2968);
  v8 = v7 + 8LL * v6;
  if ( v8 == v7 + 8LL * *(unsigned int *)(v4 + 2976) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v8 + 8LL);
}
