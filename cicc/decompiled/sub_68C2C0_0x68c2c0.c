// Function: sub_68C2C0
// Address: 0x68c2c0
//
__int64 __fastcall sub_68C2C0(_DWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  unsigned int v8; // r12d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-508h]
  __int64 v13; // [rsp+18h] [rbp-4F8h] BYREF
  _BYTE v14[160]; // [rsp+20h] [rbp-4F0h] BYREF
  __m128i v15[22]; // [rsp+C0h] [rbp-450h] BYREF
  _QWORD v16[44]; // [rsp+220h] [rbp-2F0h] BYREF
  _BYTE v17[400]; // [rsp+380h] [rbp-190h] BYREF

  v13 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  sub_6E1E00(4, v14, 0, 0);
  sub_68ACF0(a2[2], (__int64)v15);
  v6 = v15[0].m128i_i64[0];
  v7 = sub_8D4050(v15[0].m128i_i64[0]);
  if ( (unsigned int)sub_8D23E0(v6) || (unsigned int)sub_8D23B0(v7) )
  {
    v8 = 0;
    sub_685360(0x8EFu, a1, v6);
    sub_6E2B30();
  }
  else
  {
    sub_6FB570(v15);
    v10 = sub_736020(v15[0].m128i_i64[0], 0);
    a2[5] = v10;
    sub_68BC10(v10, v15);
    sub_6E2B30();
    sub_6E1E00(4, v14, 0, 0);
    sub_68ACF0(a2[2], (__int64)v15);
    sub_6FB570(v15);
    if ( (unsigned int)sub_8D4070(v6) )
    {
      v12 = sub_726700(12);
      *(_QWORD *)v12 = sub_72BA30(unk_4F06A51);
      *(_BYTE *)(v12 + 56) = 1;
      *(_QWORD *)(v12 + 64) = v6;
      sub_6E70E0(v12, v16);
      for ( ; *(_BYTE *)(v7 + 140) == 12; v7 = *(_QWORD *)(v7 + 160) )
        ;
      sub_72BAF0(v13, *(_QWORD *)(v7 + 128), unk_4F06A51);
      sub_6E6A50(v13, v17);
      sub_6F7CB0(v16, v17, 42, v16[0], v16);
    }
    else
    {
      for ( ; *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
        ;
      sub_72BAF0(v13, *(_QWORD *)(v6 + 176), unk_4F06A51);
      sub_6E6A50(v13, v16);
    }
    sub_6F7CB0(v15, v16, 50, v15[0].m128i_i64[0], v15);
    v11 = sub_736020(v15[0].m128i_i64[0], 0);
    a2[6] = v11;
    v8 = 1;
    sub_68BC10(v11, v15);
    sub_6E2B30();
  }
  sub_724E30(&v13);
  return v8;
}
