// Function: sub_145FEA0
// Address: 0x145fea0
//
__int64 __fastcall sub_145FEA0(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *i; // r12
  char v3; // al
  __int64 v4; // rax
  bool v5; // zf
  __int64 result; // rax
  void *v7; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8; // [rsp+8h] [rbp-38h] BYREF
  __int64 v9; // [rsp+18h] [rbp-28h]
  __int64 v10; // [rsp+20h] [rbp-20h]

  *(_QWORD *)(a1 + 16) = 0;
  sub_1457D90(&v7, -8, 0);
  v1 = *(_QWORD **)(a1 + 8);
  for ( i = &v1[6 * *(unsigned int *)(a1 + 24)]; v1 != i; v1 += 6 )
  {
    if ( v1 )
    {
      v3 = v8;
      v1[2] = 0;
      v1[1] = v3 & 6;
      v4 = v9;
      v5 = v9 == -8;
      v1[3] = v9;
      if ( v4 != 0 && !v5 && v4 != -16 )
        sub_1649AC0(v1 + 1, v8 & 0xFFFFFFFFFFFFFFF8LL);
      *v1 = &unk_49EC5C8;
      v1[4] = v10;
    }
  }
  v7 = &unk_49EE2B0;
  result = v9;
  if ( v9 != 0 && v9 != -8 && v9 != -16 )
    return sub_1649B30(&v8);
  return result;
}
