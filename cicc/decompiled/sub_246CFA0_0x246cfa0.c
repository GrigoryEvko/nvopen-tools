// Function: sub_246CFA0
// Address: 0x246cfa0
//
__int64 __fastcall sub_246CFA0(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 result; // rax
  _QWORD *v3; // r12
  char v4; // al
  __int64 v5; // rax
  bool v6; // zf
  _QWORD v7[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v8; // [rsp+18h] [rbp-38h]
  __int64 v9; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v7[0] = 2;
  result = *(unsigned int *)(a1 + 24);
  v7[1] = 0;
  v8 = -4096;
  v9 = 0;
  v3 = &v1[6 * result];
  if ( v3 != v1 )
  {
    do
    {
      if ( v1 )
      {
        v4 = v7[0];
        v1[2] = 0;
        v1[1] = v4 & 6;
        v5 = v8;
        v6 = v8 == 0;
        v1[3] = v8;
        if ( v5 != -4096 && !v6 && v5 != -8192 )
          sub_BD6050(v1 + 1, v7[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v1 = &unk_4A16A38;
        v1[4] = v9;
      }
      v1 += 6;
    }
    while ( v1 != v3 );
    result = v8;
    if ( v8 != -4096 && v8 != -8192 )
    {
      if ( v8 )
        return sub_BD60C0(v7);
    }
  }
  return result;
}
