// Function: sub_33CA570
// Address: 0x33ca570
//
bool __fastcall sub_33CA570(__int64 a1, void **a2)
{
  __int64 v2; // rbx
  void *v3; // r13
  __int64 v5; // rdi

  v2 = *(_QWORD *)(a1 + 96);
  v3 = *(void **)(v2 + 24);
  if ( v3 != *a2 )
    return 0;
  v5 = v2 + 24;
  if ( v3 == sub_C33340() )
    return sub_C3E590(v5, (__int64)a2);
  else
    return sub_C33D00(v5, (__int64)a2);
}
