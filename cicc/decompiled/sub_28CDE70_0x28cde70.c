// Function: sub_28CDE70
// Address: 0x28cde70
//
unsigned __int64 __fastcall sub_28CDE70(__int64 a1)
{
  __int64 v2; // [rsp+0h] [rbp-10h] BYREF
  __int64 v3; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
  v2 = *(unsigned int *)(a1 + 12);
  return sub_28CDB80(&v2, &v3, (__int64 *)(a1 + 24));
}
