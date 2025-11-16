// Function: sub_C90410
// Address: 0xc90410
//
unsigned __int64 __fastcall sub_C90410(__int64 *a1, unsigned __int64 a2, int a3)
{
  __int64 *v3; // r13
  __int64 v4; // r12
  unsigned __int64 v5; // rbx
  _QWORD v7[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( !a3 )
    a3 = sub_C8ED90(a1, a2);
  v3 = (__int64 *)(*a1 + 24LL * (unsigned int)(a3 - 1));
  v4 = (unsigned int)sub_C903C0(v3, a2);
  v5 = a2 - *(_QWORD *)(*v3 + 8);
  v7[0] = *(_QWORD *)(*v3 + 8);
  v7[1] = v5;
  return ((unsigned __int64)((unsigned int)v5 - (unsigned int)sub_C93660(v7, "\n\r", 2, -1)) << 32) | v4;
}
