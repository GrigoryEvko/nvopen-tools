// Function: sub_15A9480
// Address: 0x15a9480
//
__int64 __fastcall sub_15A9480(__int64 a1, unsigned int a2)
{
  unsigned int *v2; // rax

  v2 = (unsigned int *)sub_15A8580(a1, a2);
  if ( v2 == (unsigned int *)(*(_QWORD *)(a1 + 224) + 20LL * *(unsigned int *)(a1 + 232)) || v2[3] != a2 )
    v2 = (unsigned int *)sub_15A8580(a1, 0);
  return *v2;
}
