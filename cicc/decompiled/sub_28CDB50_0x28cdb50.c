// Function: sub_28CDB50
// Address: 0x28cdb50
//
unsigned __int64 __fastcall sub_28CDB50(__int64 a1)
{
  __int64 v2; // [rsp+0h] [rbp-10h] BYREF
  __int64 v3; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
  v2 = *(unsigned int *)(a1 + 12);
  return sub_28CD860(&v2, &v3, (__int64 *)(a1 + 24));
}
