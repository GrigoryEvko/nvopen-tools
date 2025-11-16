// Function: sub_12731E0
// Address: 0x12731e0
//
__int64 __fastcall sub_12731E0(__int64 a1, const char *a2)
{
  size_t v2; // rdx
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rax

  if ( !*(_DWORD *)(a1 + 516) )
    sub_126A910(a1);
  v2 = 0;
  if ( a2 )
    v2 = strlen(a2);
  v3 = sub_16D1B30(a1 + 504, a2, v2);
  if ( v3 == -1 )
    return 0;
  v4 = *(_QWORD *)(a1 + 504);
  v5 = v4 + 8LL * v3;
  if ( v5 == v4 + 8LL * *(unsigned int *)(a1 + 512) )
    return 0;
  else
    return *(unsigned int *)(*(_QWORD *)v5 + 8LL);
}
