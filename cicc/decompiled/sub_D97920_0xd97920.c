// Function: sub_D97920
// Address: 0xd97920
//
__int64 __fastcall sub_D97920(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned int v12; // r12d
  __int64 v13; // r14
  unsigned __int8 v15; // [rsp+6h] [rbp-2Ah] BYREF
  unsigned __int8 v16; // [rsp+7h] [rbp-29h] BYREF
  __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v17[0] = a2;
  v7 = sub_D97520(a1, v17, 1, &v15, a5, a6);
  v17[0] = a3;
  v8 = v7;
  v11 = sub_D97520(a1, v17, 1, &v16, v9, v10);
  v12 = v15;
  if ( v15 )
  {
    v12 = v16;
    if ( v16 )
    {
      v13 = v11;
      if ( v8 != v11 && !(unsigned __int8)sub_B19DB0(a1[5], v8, v11) )
        return (unsigned int)sub_B19DB0(a1[5], v13, v8);
    }
  }
  return v12;
}
