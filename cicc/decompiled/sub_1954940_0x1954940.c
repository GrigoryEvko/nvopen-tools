// Function: sub_1954940
// Address: 0x1954940
//
void *__fastcall sub_1954940(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rbx
  void *result; // rax
  _QWORD *v4; // r12
  char v5; // al
  __int64 v6; // rax
  bool v7; // zf
  _QWORD v8[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h]
  __int64 v10; // [rsp+20h] [rbp-30h]

  v1 = *(unsigned int *)(a1 + 24);
  v2 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  result = &unk_49E6B40;
  v8[0] = 2;
  v8[1] = 0;
  v4 = &v2[8 * v1];
  v9 = -8;
  v10 = 0;
  if ( v4 != v2 )
  {
    do
    {
      if ( v2 )
      {
        v5 = v8[0];
        v2[2] = 0;
        v2[1] = v5 & 6;
        v6 = v9;
        v7 = v9 == -8;
        v2[3] = v9;
        if ( v6 != 0 && !v7 && v6 != -16 )
          sub_1649AC0(v2 + 1, v8[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v2 = &unk_49E6B50;
        v2[4] = v10;
      }
      v2 += 8;
    }
    while ( v2 != v4 );
    result = (void *)v9;
    if ( v9 != -8 && v9 != -16 )
    {
      if ( v9 )
        return (void *)sub_1649B30(v8);
    }
  }
  return result;
}
