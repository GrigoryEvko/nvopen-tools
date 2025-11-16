// Function: sub_2A6AC30
// Address: 0x2a6ac30
//
__int64 __fastcall sub_2A6AC30(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 *v5; // r13
  unsigned __int8 *v6; // rax
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL);
  v3 = sub_B43CB0(a2);
  if ( sub_B2F070(v3, v2 >> 8) )
    return sub_2A6A450(a1, a2);
  v7[0] = a2;
  v5 = sub_2A686D0(a1 + 136, v7);
  v6 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)v7);
  return sub_2A63460(a1, v5, a2, v6);
}
