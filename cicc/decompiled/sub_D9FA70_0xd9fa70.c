// Function: sub_D9FA70
// Address: 0xd9fa70
//
__int64 __fastcall sub_D9FA70(__int64 a1)
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
  sub_D982A0(&v7, -4096, 0);
  v1 = *(_QWORD **)(a1 + 8);
  for ( i = &v1[6 * *(unsigned int *)(a1 + 24)]; v1 != i; v1 += 6 )
  {
    if ( v1 )
    {
      v3 = v8;
      v1[2] = 0;
      v1[1] = v3 & 6;
      v4 = v9;
      v5 = v9 == -4096;
      v1[3] = v9;
      if ( v4 != 0 && !v5 && v4 != -8192 )
        sub_BD6050(v1 + 1, v8 & 0xFFFFFFFFFFFFFFF8LL);
      *v1 = &unk_49DE910;
      v1[4] = v10;
    }
  }
  v7 = &unk_49DB368;
  result = v9;
  if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
    return sub_BD60C0(&v8);
  return result;
}
