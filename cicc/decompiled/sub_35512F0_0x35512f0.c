// Function: sub_35512F0
// Address: 0x35512f0
//
__int64 __fastcall sub_35512F0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // eax
  __int16 v16; // r13
  unsigned int v17; // r12d
  bool v18; // dl
  char v19; // al
  unsigned int v20; // edx
  unsigned int v21; // edx
  __int32 v22; // edx
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 *v25; // r12
  __int64 v26; // rax
  __int64 v27; // r15
  _QWORD *v28; // rax
  char v29; // al
  __int64 v30; // [rsp+0h] [rbp-E0h]
  __int64 v31; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v32; // [rsp+18h] [rbp-C8h]
  __int64 v33; // [rsp+20h] [rbp-C0h]
  int v34; // [rsp+34h] [rbp-ACh]
  __int64 i; // [rsp+38h] [rbp-A8h]
  _QWORD *v36; // [rsp+40h] [rbp-A0h]
  int v37; // [rsp+48h] [rbp-98h]
  unsigned int v38; // [rsp+4Ch] [rbp-94h]
  unsigned __int8 *v39; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int8 *v40; // [rsp+58h] [rbp-88h] BYREF
  unsigned __int8 *v41; // [rsp+60h] [rbp-80h] BYREF
  __int64 v42; // [rsp+68h] [rbp-78h]
  __int64 v43; // [rsp+70h] [rbp-70h]
  __m128i v44; // [rsp+80h] [rbp-60h] BYREF
  __int64 v45; // [rsp+90h] [rbp-50h]
  __int64 v46; // [rsp+98h] [rbp-48h]
  __int64 v47; // [rsp+A0h] [rbp-40h]

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_52:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501EACC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_52;
  }
  v33 = *(_QWORD *)(a1[25] + 32LL);
  v31 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                      *(_QWORD *)(v3 + 8),
                      &unk_501EACC)
                  + 232);
  result = sub_2E311E0(a2);
  v8 = *(_QWORD *)(a2 + 56);
  v30 = result;
  if ( v8 == result )
    return result;
  do
  {
    v9 = *(_QWORD *)(v8 + 32);
    v32 = *(_QWORD *)(*(_QWORD *)(v33 + 56) + 16LL * (*(_DWORD *)(v9 + 8) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    result = *(_DWORD *)(v8 + 40) & 0xFFFFFF;
    v34 = result;
    if ( (_DWORD)result == 1 )
      goto LABEL_46;
    v38 = 1;
    for ( i = v8; ; v9 = *(_QWORD *)(i + 32) )
    {
      v23 = v9 + 40LL * v38;
      if ( (*(_DWORD *)v23 & 0xFFF00) == 0 )
        goto LABEL_36;
      v37 = sub_2EC06C0(v33, v32, byte_3F871B3, 0, v6, v7);
      v24 = *(_QWORD *)(*(_QWORD *)(i + 32) + 40LL * (v38 + 1) + 24);
      v25 = (__int64 *)sub_2E313E0(v24);
      sub_2E32810((__int64 *)&v39, v24, (__int64)v25);
      v26 = a1[30];
      v40 = v39;
      v27 = *(_QWORD *)(v26 + 8) - 800LL;
      if ( !v39 )
      {
        v41 = 0;
LABEL_41:
        v28 = *(_QWORD **)(v24 + 32);
        v42 = 0;
        v43 = 0;
        v36 = v28;
        v44.m128i_i64[0] = 0;
        v11 = sub_2E7B380(v28, v27, (unsigned __int8 **)&v44, 0);
        goto LABEL_12;
      }
      sub_B96E90((__int64)&v40, (__int64)v39, 1);
      v41 = v40;
      if ( !v40 )
        goto LABEL_41;
      sub_B976B0((__int64)&v40, v40, (__int64)&v41);
      v10 = *(_QWORD **)(v24 + 32);
      v40 = 0;
      v42 = 0;
      v43 = 0;
      v36 = v10;
      v44.m128i_i64[0] = (__int64)v41;
      if ( v41 )
        sub_B96E90((__int64)&v44, (__int64)v41, 1);
      v11 = sub_2E7B380(v36, v27, (unsigned __int8 **)&v44, 0);
LABEL_12:
      v12 = (unsigned __int64)v11;
      if ( v44.m128i_i64[0] )
        sub_B91220((__int64)&v44, v44.m128i_i64[0]);
      sub_2E31040((__int64 *)(v24 + 40), v12);
      v13 = *v25;
      v14 = *(_QWORD *)v12;
      *(_QWORD *)(v12 + 8) = v25;
      v13 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v12 = v13 | v14 & 7;
      *(_QWORD *)(v13 + 8) = v12;
      *v25 = v12 | *v25 & 7;
      if ( v42 )
        sub_2E882B0(v12, (__int64)v36, v42);
      if ( v43 )
        sub_2E88680(v12, (__int64)v36, v43);
      v44.m128i_i64[0] = 0x10000000;
      v44.m128i_i32[2] = v37;
      v45 = 0;
      v46 = 0;
      v47 = 0;
      sub_2E8EAD0(v12, (__int64)v36, &v44);
      v15 = *(unsigned __int8 *)(v23 + 3);
      v16 = (*(_DWORD *)v23 >> 8) & 0xFFF;
      if ( (v15 & 0x10) != 0 )
      {
        v18 = (v15 & 0x40) != 0;
        v17 = (v15 & 0x20) == 0 ? 2 : 6;
LABEL_21:
        if ( v18 && (v15 & 0x10) != 0 )
          v17 |= 0x10u;
        goto LABEL_23;
      }
      v17 = (v15 >> 3) & 4;
      v18 = (v15 & 0x40) != 0;
      if ( (v15 & 0x40) != 0 )
      {
        v17 |= 8u;
        goto LABEL_21;
      }
LABEL_23:
      v19 = *(_BYTE *)(v23 + 4);
      if ( (v19 & 1) != 0 )
        v17 |= 0x20u;
      v20 = v17;
      if ( (v19 & 2) != 0 )
      {
        BYTE1(v20) = BYTE1(v17) | 1;
        v17 = v20;
      }
      v21 = v17;
      if ( (v19 & 8) != 0 )
      {
        LOBYTE(v21) = v17 | 0x80;
        v17 = v21;
      }
      v22 = *(_DWORD *)(v23 + 8);
      if ( (unsigned int)(v22 - 1) <= 0x3FFFFFFE )
      {
        v29 = sub_2EAB300(v23);
        v22 = *(_DWORD *)(v23 + 8);
        if ( v29 )
          v17 |= 0x200u;
      }
      v44.m128i_i8[0] = 0;
      v44.m128i_i32[2] = v22;
      v45 = 0;
      v46 = 0;
      v47 = 0;
      v44.m128i_i8[3] = ((unsigned __int8)(v17 >> 9) << 7)
                      | (((v17 & 0x18) != 0) << 6)
                      | (32 * ((v17 & 4) != 0)) & 0x3F
                      | (16 * ((v17 & 2) != 0)) & 0x3F
                      | v44.m128i_i8[3] & 0xF;
      v44.m128i_i16[1] &= 0xF00Fu;
      v44.m128i_i8[4] = (8 * ((unsigned __int8)v17 >> 7))
                      | (2 * (BYTE1(v17) & 1)) & 0xF3
                      | ((v17 & 0x20) != 0)
                      | v44.m128i_i8[4] & 0xF0;
      v44.m128i_i32[0] = ((v16 & 0xFFF) << 8) | v44.m128i_i32[0] & 0xFFF000FF;
      sub_2E8EAD0(v12, (__int64)v36, &v44);
      if ( v41 )
        sub_B91220((__int64)&v41, (__int64)v41);
      if ( v40 )
        sub_B91220((__int64)&v40, (__int64)v40);
      sub_2E192D0(v31, v12, 0);
      sub_2EAB0C0(v23, v37);
      *(_DWORD *)v23 &= 0xFFF000FF;
      if ( v39 )
        sub_B91220((__int64)&v39, (__int64)v39);
LABEL_36:
      v38 += 2;
      result = v38;
      if ( v34 == v38 )
        break;
    }
    v8 = i;
LABEL_46:
    if ( (*(_BYTE *)v8 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
        v8 = *(_QWORD *)(v8 + 8);
    }
    v8 = *(_QWORD *)(v8 + 8);
  }
  while ( v30 != v8 );
  return result;
}
