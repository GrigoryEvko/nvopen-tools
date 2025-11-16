// Function: sub_6C6630
// Address: 0x6c6630
//
__int64 __fastcall sub_6C6630(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 *v5; // r13
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 *v8; // r15
  __int64 v9; // rsi
  __int64 v10; // r13
  __int64 v12; // [rsp+10h] [rbp-130h]
  __int64 v13; // [rsp+18h] [rbp-128h]
  char v14; // [rsp+37h] [rbp-109h] BYREF
  __int64 v15; // [rsp+38h] [rbp-108h] BYREF
  _OWORD v16[2]; // [rsp+40h] [rbp-100h] BYREF
  __int128 v17; // [rsp+60h] [rbp-E0h]
  _BYTE v18[208]; // [rsp+70h] [rbp-D0h] BYREF

  v2 = *(_QWORD *)(a1 + 360);
  v15 = 0;
  v13 = v2;
  v3 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL);
  memset(v16, 0, sizeof(v16));
  v12 = v3;
  v17 = 0;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE10(v17) |= 1u;
  DWORD2(v17) |= 0x10000009u;
  v4 = *(_QWORD *)(unk_4F04C50 + 40LL);
  if ( v4 )
  {
    v5 = &v15;
    do
    {
      v6 = *(_QWORD *)(v4 + 120);
      v14 = 0;
      sub_6E1E00(4, v18, 0, 0);
      v7 = sub_6E2F40(0);
      *v5 = v7;
      v8 = (__int64 *)(*(_QWORD *)(v7 + 24) + 8LL);
      sub_6F8E70(v4, dword_4F07508, dword_4F07508, v8, 0);
      if ( !(unsigned int)sub_8D32E0(v6) )
        v6 = sub_72D6A0(v6);
      sub_6BFEC0(v6, v8, dword_4F07508, dword_4F07508, &v14);
      sub_6E2B30(v6, v8);
      v4 = *(_QWORD *)(v4 + 112);
      v5 = (__int64 *)*v5;
    }
    while ( v4 );
    v4 = v15;
  }
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  sub_6C64D0(v12, v9, v13 + 8, dword_4D048B8, 1u, v4, (__int64)v16);
  v10 = *((_QWORD *)&v16[0] + 1);
  sub_6E1990(v4);
  if ( !v10 )
    return sub_72C9D0(v4, v9);
  return v10;
}
