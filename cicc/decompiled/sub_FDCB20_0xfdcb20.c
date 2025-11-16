// Function: sub_FDCB20
// Address: 0xfdcb20
//
__int64 __fastcall sub_FDCB20(unsigned __int64 a1, unsigned __int16 a2, unsigned __int64 a3, __int16 a4)
{
  unsigned __int64 v6; // r12
  __int16 v8; // r14
  __int16 v9; // bx
  unsigned __int64 v10; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int16 v11[2]; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int16 v12; // [rsp+14h] [rbp-2Ch] BYREF
  unsigned __int64 v13[5]; // [rsp+18h] [rbp-28h] BYREF

  v13[0] = a1;
  v10 = a3;
  v12 = a2;
  v11[0] = a4;
  sub_FDCA70(v13, &v12, &v10, v11);
  v6 = v13[0];
  if ( v13[0] <= v10 )
    return 0;
  v8 = v12;
  if ( v10 || !a3 )
    return v13[0] - v10;
  v9 = sub_D788C0(a3, a4);
  if ( (unsigned int)sub_D788E0(v6, v8, 1u, v9 + 64) )
    return v13[0];
  else
    return -1;
}
