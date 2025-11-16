// Function: sub_26BDAB0
// Address: 0x26bdab0
//
__int64 __fastcall sub_26BDAB0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r13d
  __int64 v4; // r14
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i v9; // [rsp+0h] [rbp-E0h] BYREF
  const char *v10; // [rsp+10h] [rbp-D0h]
  __int64 v11; // [rsp+18h] [rbp-C8h]
  __int16 v12; // [rsp+20h] [rbp-C0h]
  __m128i v13[2]; // [rsp+30h] [rbp-B0h] BYREF
  char v14; // [rsp+50h] [rbp-90h]
  char v15; // [rsp+51h] [rbp-8Fh]
  __m128i v16[3]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v17[4]; // [rsp+90h] [rbp-50h] BYREF
  int v18; // [rsp+B0h] [rbp-30h]
  __m128i *v19; // [rsp+B8h] [rbp-28h]

  v1 = sub_B92180(a1);
  if ( v1 )
  {
    return *(unsigned int *)(v1 + 16);
  }
  else
  {
    v2 = 0;
    if ( !LOBYTE(qword_500BB08[8]) )
    {
      v15 = 1;
      v4 = sub_B2BE50(a1);
      v14 = 3;
      v13[0].m128i_i64[0] = (__int64)": Function profile not used";
      v5 = sub_BD5D20(a1);
      v12 = 1283;
      v11 = v6;
      v10 = v5;
      v9.m128i_i64[0] = (__int64)"No debug information found in function ";
      sub_9C6370(v16, &v9, v13, (__int64)"No debug information found in function ", v7, v8);
      v19 = v16;
      v17[2] = 0;
      v17[1] = 0x10000000CLL;
      v17[3] = 0;
      v18 = 0;
      v17[0] = &unk_49D9C78;
      sub_B6EB20(v4, (__int64)v17);
    }
  }
  return v2;
}
