// Function: sub_913450
// Address: 0x913450
//
__int64 __fastcall sub_913450(__int64 a1, const char *a2)
{
  size_t v2; // r13
  unsigned int v3; // eax
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rax

  if ( !*(_DWORD *)(a1 + 492) )
    sub_90AEE0(a1);
  v2 = 0;
  if ( a2 )
    v2 = strlen(a2);
  v3 = sub_C92610(a2, v2);
  v4 = sub_C92860(a1 + 480, a2, v2, v3);
  if ( v4 == -1 )
    return 0;
  v5 = *(_QWORD *)(a1 + 480);
  v6 = v5 + 8LL * v4;
  if ( v6 == v5 + 8LL * *(unsigned int *)(a1 + 488) )
    return 0;
  else
    return *(unsigned int *)(*(_QWORD *)v6 + 8LL);
}
