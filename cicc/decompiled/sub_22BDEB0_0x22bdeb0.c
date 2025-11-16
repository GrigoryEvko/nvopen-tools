// Function: sub_22BDEB0
// Address: 0x22bdeb0
//
__int64 __fastcall sub_22BDEB0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  __int64 v3; // r12
  char v4; // al
  __int64 v5; // rax
  bool v6; // zf
  _QWORD v7[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v8; // [rsp+18h] [rbp-38h]
  unsigned __int8 v9; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v7[0] = 2;
  result = *(unsigned int *)(a1 + 24);
  v7[1] = 0;
  v9 = 0;
  v8 = -4096;
  v3 = v1 + 48 * result;
  if ( v3 != v1 )
  {
    do
    {
      if ( v1 )
      {
        v4 = v7[0];
        *(_QWORD *)(v1 + 16) = 0;
        *(_QWORD *)(v1 + 8) = v4 & 6;
        v5 = v8;
        v6 = v8 == 0;
        *(_QWORD *)(v1 + 24) = v8;
        if ( v5 != -4096 && !v6 && v5 != -8192 )
          sub_BD6050((unsigned __int64 *)(v1 + 8), v7[0] & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)v1 = &unk_49DE8C0;
        result = v9;
        *(_BYTE *)(v1 + 32) = v9;
      }
      v1 += 48;
    }
    while ( v1 != v3 );
    if ( !v9 )
    {
      result = v8;
      if ( v8 != -8192 && v8 != 0 && v8 != -4096 )
        return sub_BD60C0(v7);
    }
  }
  return result;
}
