// Function: sub_1490810
// Address: 0x1490810
//
__int64 __fastcall sub_1490810(__int64 *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v5; // r12
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // r14
  __int64 *v13; // rdi
  _QWORD *v14; // rdx
  _QWORD *v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rdx
  _QWORD *v18; // r14
  __int64 v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // r14
  __int64 *v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // rdx
  _QWORD *v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-80h]
  _QWORD *v29; // [rsp+0h] [rbp-80h]
  _QWORD *v30; // [rsp+0h] [rbp-80h]
  _QWORD *v31; // [rsp+0h] [rbp-80h]
  _QWORD *v32; // [rsp+0h] [rbp-80h]
  char v33; // [rsp+17h] [rbp-69h]
  char v34; // [rsp+17h] [rbp-69h]
  char v35; // [rsp+17h] [rbp-69h]
  char v36; // [rsp+17h] [rbp-69h]
  char v37; // [rsp+17h] [rbp-69h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+18h] [rbp-68h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v44; // [rsp+30h] [rbp-50h] BYREF
  __int64 v45; // [rsp+38h] [rbp-48h]
  _BYTE v46[64]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a2;
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
      return sub_145CF40(*a1, *(_QWORD *)(a2 + 32) + 24LL);
    case 1:
      v7 = sub_1490D40(a1, *(_QWORD *)(a2 + 32));
      if ( v7 == *(_QWORD *)(v5 + 32) )
        return v5;
      return sub_14835F0((_QWORD *)*a1, v7, *(_QWORD *)(v5 + 40), 0, a3, a4);
    case 2:
      v8 = sub_1490D40(a1, *(_QWORD *)(a2 + 32));
      if ( v8 == *(_QWORD *)(v5 + 32) )
        return v5;
      return sub_14747F0(*a1, v8, *(_QWORD *)(v5 + 40), 0);
    case 3:
      v23 = sub_1490D40(a1, *(_QWORD *)(a2 + 32));
      if ( v23 == *(_QWORD *)(v5 + 32) )
        return v5;
      return sub_147B0D0(*a1, v23, *(_QWORD *)(v5 + 40), 0);
    case 4:
      v20 = *(_QWORD **)(a2 + 32);
      v44 = (__int64 *)v46;
      v45 = 0x200000000LL;
      v31 = &v20[*(_QWORD *)(a2 + 40)];
      if ( v20 == v31 )
        return v5;
      v36 = 0;
      v21 = v20;
      do
      {
        v41 = *v21;
        v43 = sub_1490D40(a1, *v21);
        sub_1458920((__int64)&v44, &v43);
        v13 = v44;
        ++v21;
        v36 |= v44[(unsigned int)v45 - 1] != v41;
      }
      while ( v31 != v21 );
      if ( v36 )
      {
        v22 = sub_147DD40(*a1, (__int64 *)&v44, 0, 0, a3, a4);
        v13 = v44;
        v5 = (__int64)v22;
      }
      goto LABEL_16;
    case 5:
      v24 = *(_QWORD **)(a2 + 32);
      v44 = (__int64 *)v46;
      v45 = 0x200000000LL;
      v32 = &v24[*(_QWORD *)(a2 + 40)];
      if ( v24 == v32 )
        return v5;
      v37 = 0;
      v25 = v24;
      do
      {
        v42 = *v25;
        v43 = sub_1490D40(a1, *v25);
        sub_1458920((__int64)&v44, &v43);
        v13 = v44;
        ++v25;
        v37 |= v44[(unsigned int)v45 - 1] != v42;
      }
      while ( v32 != v25 );
      if ( v37 )
      {
        v26 = sub_147EE30((_QWORD *)*a1, &v44, 0, 0, a3, a4);
        v13 = v44;
        v5 = v26;
      }
      goto LABEL_16;
    case 6:
      v9 = sub_1490D40(a1, *(_QWORD *)(a2 + 32));
      v10 = sub_1490D40(a1, *(_QWORD *)(a2 + 40));
      if ( v9 == *(_QWORD *)(a2 + 32) && v10 == *(_QWORD *)(a2 + 40) )
        return v5;
      return sub_1483CF0((_QWORD *)*a1, v9, v10, a3, a4);
    case 7:
      v11 = *(_QWORD **)(a2 + 32);
      v44 = (__int64 *)v46;
      v45 = 0x200000000LL;
      v28 = &v11[*(_QWORD *)(a2 + 40)];
      if ( v11 == v28 )
        return v5;
      v33 = 0;
      v12 = v11;
      do
      {
        v38 = *v12;
        v43 = sub_1490D40(a1, *v12);
        sub_1458920((__int64)&v44, &v43);
        v13 = v44;
        ++v12;
        v33 |= v44[(unsigned int)v45 - 1] != v38;
      }
      while ( v28 != v12 );
      if ( v33 )
      {
        v27 = sub_14785F0(*a1, &v44, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 26) & 7);
        v13 = v44;
        v5 = v27;
      }
      goto LABEL_16;
    case 8:
      v14 = *(_QWORD **)(a2 + 32);
      v44 = (__int64 *)v46;
      v45 = 0x200000000LL;
      v29 = &v14[*(_QWORD *)(a2 + 40)];
      if ( v14 == v29 )
        return v5;
      v34 = 0;
      v15 = v14;
      do
      {
        v39 = *v15;
        v43 = sub_1490D40(a1, *v15);
        sub_1458920((__int64)&v44, &v43);
        v13 = v44;
        ++v15;
        v34 |= v44[(unsigned int)v45 - 1] != v39;
      }
      while ( v29 != v15 );
      if ( v34 )
      {
        v16 = sub_14813B0((_QWORD *)*a1, &v44, a3, a4);
        v13 = v44;
        v5 = v16;
      }
      goto LABEL_16;
    case 9:
      v17 = *(_QWORD **)(a2 + 32);
      v44 = (__int64 *)v46;
      v45 = 0x200000000LL;
      v30 = &v17[*(_QWORD *)(a2 + 40)];
      if ( v17 == v30 )
        return v5;
      v35 = 0;
      v18 = v17;
      do
      {
        v40 = *v18;
        v43 = sub_1490D40(a1, *v18);
        sub_1458920((__int64)&v44, &v43);
        v13 = v44;
        ++v18;
        v35 |= v44[(unsigned int)v45 - 1] != v40;
      }
      while ( v30 != v18 );
      if ( v35 )
      {
        v19 = sub_147A3C0((_QWORD *)*a1, &v44, a3, a4);
        v13 = v44;
        v5 = v19;
      }
LABEL_16:
      if ( v13 != (__int64 *)v46 )
        _libc_free((unsigned __int64)v13);
      return v5;
    case 0xA:
      return sub_145DC80(*a1, *(_QWORD *)(a2 - 8));
    case 0xB:
      return sub_1456E90(*a1);
  }
}
