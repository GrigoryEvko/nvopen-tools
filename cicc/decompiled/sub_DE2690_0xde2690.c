// Function: sub_DE2690
// Address: 0xde2690
//
__int64 __fastcall sub_DE2690(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 *v5; // rax
  bool v6; // zf
  _QWORD v7[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v8; // [rsp+10h] [rbp-70h]
  __int64 v9; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v10; // [rsp+20h] [rbp-60h]
  __int64 v11; // [rsp+58h] [rbp-28h] BYREF
  int v12; // [rsp+60h] [rbp-20h]

  v2 = a2;
  if ( !*(_DWORD *)(a1 + 16) )
    return v2;
  v4 = *(_QWORD *)(a1 + 40);
  v7[1] = 0;
  v8 = 1;
  v7[0] = v4;
  v5 = &v9;
  do
  {
    *v5 = -4096;
    v5 += 2;
  }
  while ( v5 != &v11 );
  v6 = *(_BYTE *)(a1 + 32) == 0;
  v11 = a1;
  v12 = 0;
  if ( !v6 )
    v12 = 2;
  if ( *(_BYTE *)(a1 + 33) )
    v12 |= 4u;
  v2 = sub_DE1A30((__int64)v7, a2);
  if ( (v8 & 1) != 0 )
    return v2;
  sub_C7D6A0(v9, 16LL * v10, 8);
  return v2;
}
