// Function: sub_18A57D0
// Address: 0x18a57d0
//
__int64 __fastcall sub_18A57D0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r13d
  __int64 v4; // r14
  __int64 v5; // rdx
  _QWORD v6[2]; // [rsp+0h] [rbp-C0h] BYREF
  __m128i v7[2]; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v8; // [rsp+30h] [rbp-90h] BYREF
  char v9; // [rsp+40h] [rbp-80h]
  char v10; // [rsp+41h] [rbp-7Fh]
  __m128i v11; // [rsp+50h] [rbp-70h] BYREF
  __int16 v12; // [rsp+60h] [rbp-60h]
  _QWORD v13[4]; // [rsp+70h] [rbp-50h] BYREF
  int v14; // [rsp+90h] [rbp-30h]
  __m128i *v15; // [rsp+98h] [rbp-28h]

  v1 = sub_1626D20(a1);
  if ( v1 )
  {
    return *(unsigned int *)(v1 + 24);
  }
  else
  {
    v2 = 0;
    if ( !byte_4FAD2C0 )
    {
      v10 = 1;
      v4 = sub_15E0530(a1);
      v9 = 3;
      v8.m128i_i64[0] = (__int64)": Function profile not used";
      v6[0] = sub_1649960(a1);
      v12 = 1283;
      v6[1] = v5;
      v11.m128i_i64[0] = (__int64)"No debug information found in function ";
      v11.m128i_i64[1] = (__int64)v6;
      sub_14EC200(v7, &v11, &v8);
      v15 = v7;
      v13[2] = 0;
      v13[1] = 0x100000007LL;
      v13[3] = 0;
      v14 = 0;
      v13[0] = &unk_49ECF18;
      sub_16027F0(v4, (__int64)v13);
    }
  }
  return v2;
}
