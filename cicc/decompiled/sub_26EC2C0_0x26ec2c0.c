// Function: sub_26EC2C0
// Address: 0x26ec2c0
//
void __fastcall sub_26EC2C0(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  float *v8; // rax
  float v10; // [rsp+Ch] [rbp-64h]
  __m128i v11; // [rsp+10h] [rbp-60h] BYREF
  _DWORD v12[5]; // [rsp+20h] [rbp-50h] BYREF
  char v13; // [rsp+34h] [rbp-3Ch]

  v3 = a2 + 48;
  v4 = *(_QWORD *)(a2 + 56);
  if ( v4 != a2 + 48 )
  {
    do
    {
      while ( 1 )
      {
        v5 = v4 - 24;
        if ( !v4 )
          v5 = 0;
        v6 = v5;
        sub_3143F80(v12, v5, a3);
        if ( v13 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          return;
      }
      v7 = sub_B10CD0(v6 + 48);
      v11.m128i_i64[1] = sub_26E9470(v7);
      v11.m128i_i64[0] = v12[0];
      v10 = *(float *)&v12[4];
      v8 = (float *)sub_26EC210(a3, &v11);
      *v8 = v10 + *v8;
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v3 != v4 );
  }
}
