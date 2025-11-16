// Function: sub_22BDDA0
// Address: 0x22bdda0
//
__int64 __fastcall sub_22BDDA0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rbx
  __int64 result; // rax
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
  result = 5 * v1;
  v8[0] = 2;
  v4 = &v2[result];
  v8[1] = 0;
  v9 = -4096;
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
        v7 = v9 == 0;
        v2[3] = v9;
        if ( v6 != -4096 && !v7 && v6 != -8192 )
          sub_BD6050(v2 + 1, v8[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v2 = off_4A09D90;
        v2[4] = v10;
      }
      v2 += 5;
    }
    while ( v2 != v4 );
    result = v9;
    if ( v9 != -4096 && v9 != -8192 )
    {
      if ( v9 )
        return sub_BD60C0(v8);
    }
  }
  return result;
}
