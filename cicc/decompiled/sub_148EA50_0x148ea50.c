// Function: sub_148EA50
// Address: 0x148ea50
//
__int64 __fastcall sub_148EA50(__int64 *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v5; // rbx
  _QWORD *v6; // rdx
  _QWORD *v7; // r14
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 result; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdi
  char v15; // r9
  __int64 v16; // rbx
  char v17; // al
  __int64 *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rsi
  _QWORD *v24; // rdx
  _QWORD *v25; // r14
  __int64 *v26; // rax
  _QWORD *v27; // rdx
  _QWORD *v28; // r14
  __int64 v29; // rax
  _QWORD *v30; // rdx
  _QWORD *v31; // r14
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rax
  unsigned int v35; // esi
  int v36; // eax
  int v37; // eax
  __int64 v38; // rax
  _QWORD *v39; // [rsp+0h] [rbp-80h]
  _QWORD *v40; // [rsp+0h] [rbp-80h]
  _QWORD *v41; // [rsp+0h] [rbp-80h]
  _QWORD *v42; // [rsp+0h] [rbp-80h]
  _QWORD *v43; // [rsp+0h] [rbp-80h]
  char v44; // [rsp+17h] [rbp-69h]
  char v45; // [rsp+17h] [rbp-69h]
  char v46; // [rsp+17h] [rbp-69h]
  char v47; // [rsp+17h] [rbp-69h]
  char v48; // [rsp+17h] [rbp-69h]
  __int64 v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+18h] [rbp-68h]
  __int64 v51; // [rsp+18h] [rbp-68h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+20h] [rbp-60h] BYREF
  __int64 v55; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v56; // [rsp+30h] [rbp-50h] BYREF
  __int64 v57; // [rsp+38h] [rbp-48h]
  _BYTE v58[64]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a2;
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
    case 0xB:
      return v5;
    case 1:
      v21 = sub_148F000(a1, *(_QWORD *)(a2 + 32));
      if ( v21 == *(_QWORD *)(v5 + 32) )
        return v5;
      return sub_14835F0((_QWORD *)*a1, v21, *(_QWORD *)(v5 + 40), 0, a3, a4);
    case 2:
      v22 = sub_148F000(a1, *(_QWORD *)(a2 + 32));
      if ( v22 == *(_QWORD *)(v5 + 32) )
        return v5;
      return sub_14747F0(*a1, v22, *(_QWORD *)(v5 + 40), 0);
    case 3:
      v23 = sub_148F000(a1, *(_QWORD *)(a2 + 32));
      if ( v23 == *(_QWORD *)(v5 + 32) )
        return v5;
      return sub_147B0D0(*a1, v23, *(_QWORD *)(v5 + 40), 0);
    case 4:
      v24 = *(_QWORD **)(a2 + 32);
      v56 = (__int64 *)v58;
      v57 = 0x200000000LL;
      v41 = &v24[*(_QWORD *)(a2 + 40)];
      if ( v24 == v41 )
        return v5;
      v46 = 0;
      v25 = v24;
      do
      {
        v51 = *v25;
        v55 = sub_148F000(a1, *v25);
        sub_1458920((__int64)&v56, &v55);
        v8 = v56;
        ++v25;
        v46 |= v56[(unsigned int)v57 - 1] != v51;
      }
      while ( v41 != v25 );
      if ( v46 )
      {
        v26 = sub_147DD40(*a1, (__int64 *)&v56, 0, 0, a3, a4);
        v8 = v56;
        v5 = (__int64)v26;
      }
      goto LABEL_7;
    case 5:
      v27 = *(_QWORD **)(a2 + 32);
      v56 = (__int64 *)v58;
      v57 = 0x200000000LL;
      v42 = &v27[*(_QWORD *)(a2 + 40)];
      if ( v27 == v42 )
        return v5;
      v47 = 0;
      v28 = v27;
      do
      {
        v52 = *v28;
        v55 = sub_148F000(a1, *v28);
        sub_1458920((__int64)&v56, &v55);
        v8 = v56;
        ++v28;
        v47 |= v56[(unsigned int)v57 - 1] != v52;
      }
      while ( v42 != v28 );
      if ( v47 )
      {
        v29 = sub_147EE30((_QWORD *)*a1, &v56, 0, 0, a3, a4);
        v8 = v56;
        v5 = v29;
      }
      goto LABEL_7;
    case 6:
      v33 = sub_148F000(a1, *(_QWORD *)(a2 + 32));
      v34 = sub_148F000(a1, *(_QWORD *)(a2 + 40));
      if ( v33 == *(_QWORD *)(a2 + 32) && v34 == *(_QWORD *)(a2 + 40) )
        return v5;
      return sub_1483CF0((_QWORD *)*a1, v33, v34, a3, a4);
    case 7:
      v30 = *(_QWORD **)(a2 + 32);
      v56 = (__int64 *)v58;
      v57 = 0x200000000LL;
      v43 = &v30[*(_QWORD *)(a2 + 40)];
      if ( v30 == v43 )
        return v5;
      v48 = 0;
      v31 = v30;
      do
      {
        v53 = *v31;
        v55 = sub_148F000(a1, *v31);
        sub_1458920((__int64)&v56, &v55);
        v8 = v56;
        ++v31;
        v48 |= v56[(unsigned int)v57 - 1] != v53;
      }
      while ( v43 != v31 );
      if ( v48 )
      {
        v32 = sub_14785F0(*a1, &v56, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 26) & 7);
        v8 = v56;
        v5 = v32;
      }
      goto LABEL_7;
    case 8:
      v6 = *(_QWORD **)(a2 + 32);
      v56 = (__int64 *)v58;
      v57 = 0x200000000LL;
      v39 = &v6[*(_QWORD *)(a2 + 40)];
      if ( v6 == v39 )
        return v5;
      v44 = 0;
      v7 = v6;
      do
      {
        v49 = *v7;
        v55 = sub_148F000(a1, *v7);
        sub_1458920((__int64)&v56, &v55);
        v8 = v56;
        ++v7;
        v44 |= v56[(unsigned int)v57 - 1] != v49;
      }
      while ( v39 != v7 );
      if ( v44 )
      {
        v9 = sub_14813B0((_QWORD *)*a1, &v56, a3, a4);
        v8 = v56;
        v5 = v9;
      }
      goto LABEL_7;
    case 9:
      v11 = *(_QWORD **)(a2 + 32);
      v56 = (__int64 *)v58;
      v57 = 0x200000000LL;
      v40 = &v11[*(_QWORD *)(a2 + 40)];
      if ( v11 == v40 )
        return v5;
      v45 = 0;
      v12 = v11;
      do
      {
        v50 = *v12;
        v55 = sub_148F000(a1, *v12);
        sub_1458920((__int64)&v56, &v55);
        v8 = v56;
        ++v12;
        v45 |= v56[(unsigned int)v57 - 1] != v50;
      }
      while ( v40 != v12 );
      if ( v45 )
      {
        v13 = sub_147A3C0((_QWORD *)*a1, &v56, a3, a4);
        v8 = v56;
        v5 = v13;
      }
LABEL_7:
      if ( v8 != (__int64 *)v58 )
        _libc_free((unsigned __int64)v8);
      return v5;
    case 0xA:
      v14 = a1[5];
      v54 = *(_QWORD *)(a2 - 8);
      v55 = v54;
      v15 = sub_145CB40(v14, &v55, &v56);
      result = a2;
      if ( !v15 )
        return result;
      v16 = a1[5];
      v17 = sub_145CB40(v16, &v54, &v56);
      v18 = v56;
      if ( v17 )
      {
        v19 = v56[1];
        goto LABEL_19;
      }
      v35 = *(_DWORD *)(v16 + 24);
      v36 = *(_DWORD *)(v16 + 16);
      ++*(_QWORD *)v16;
      v37 = v36 + 1;
      if ( 4 * v37 >= 3 * v35 )
      {
        v35 *= 2;
      }
      else if ( v35 - *(_DWORD *)(v16 + 20) - v37 > v35 >> 3 )
      {
        goto LABEL_49;
      }
      sub_14669A0(v16, v35);
      sub_145CB40(v16, &v54, &v56);
      v18 = v56;
      v37 = *(_DWORD *)(v16 + 16) + 1;
LABEL_49:
      *(_DWORD *)(v16 + 16) = v37;
      if ( *v18 != -8 )
        --*(_DWORD *)(v16 + 20);
      v38 = v54;
      v18[1] = 0;
      v19 = 0;
      *v18 = v38;
LABEL_19:
      v20 = *a1;
      if ( *((_BYTE *)a1 + 48) && *(_BYTE *)(v19 + 16) == 13 )
        return sub_145CE20(v20, v19);
      else
        return sub_145DC80(v20, v19);
  }
}
