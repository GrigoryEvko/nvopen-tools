// Function: sub_19D6020
// Address: 0x19d6020
//
__int64 __fastcall sub_19D6020(__int128 a1)
{
  unsigned int v1; // r13d
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // eax
  __m128i *v13; // r8
  __int64 v14; // rcx
  __int64 v15; // [rsp-8h] [rbp-A8h]
  __m128i v16; // [rsp+10h] [rbp-90h] BYREF
  void (__fastcall *v17)(__m128i *, __m128i *, __int64, __int64, __m128i *); // [rsp+20h] [rbp-80h]
  __int64 (__fastcall *v18)(__int64); // [rsp+28h] [rbp-78h]
  __int128 v19; // [rsp+30h] [rbp-70h] BYREF
  __int64 (__fastcall *v20)(const __m128i **, const __m128i *, int); // [rsp+40h] [rbp-60h]
  __int64 (__fastcall *v21)(__int64 *); // [rsp+48h] [rbp-58h]
  __m128i v22; // [rsp+50h] [rbp-50h] BYREF
  __int64 (__fastcall *v23)(__m128i *, __m128i *, int); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v24)(__int64); // [rsp+68h] [rbp-38h]

  v1 = 0;
  if ( !(unsigned __int8)sub_1636880(a1, *((__int64 *)&a1 + 1)) )
  {
    v3 = *(__int64 **)(a1 + 8);
    v4 = *v3;
    v5 = v3[1];
    if ( v4 == v5 )
LABEL_20:
      BUG();
    while ( *(_UNKNOWN **)v4 != &unk_4F99308 )
    {
      v4 += 16;
      if ( v5 == v4 )
        goto LABEL_20;
    }
    v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F99308);
    v7 = *(__int64 **)(a1 + 8);
    v8 = v6 + 160;
    v9 = *v7;
    v10 = v7[1];
    if ( v9 == v10 )
LABEL_21:
      BUG();
    while ( *(_UNKNOWN **)v9 != &unk_4F9B6E8 )
    {
      v9 += 16;
      if ( v10 == v9 )
        goto LABEL_21;
    }
    v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9B6E8);
    v18 = sub_19CEC70;
    v17 = (void (__fastcall *)(__m128i *, __m128i *, __int64, __int64, __m128i *))sub_19CEA70;
    v21 = sub_19CECC0;
    v20 = sub_19CEAA0;
    v24 = sub_19CED20;
    v23 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_19CEAD0;
    v16.m128i_i64[0] = a1;
    v19 = a1;
    v22.m128i_i64[0] = a1;
    v12 = sub_19D5E90((__m128i *)(a1 + 160), *((__int64 *)&a1 + 1), v8, v11 + 360, &v22, (__m128i *)&v19, &v16);
    v13 = &v22;
    v1 = v12;
    v14 = v15;
    if ( v23 )
      v23(&v22, &v22, 3);
    if ( v20 )
      ((void (__fastcall *)(__int128 *, __int128 *, __int64, __int64, __m128i *))v20)(&v19, &v19, 3, v14, v13);
    if ( v17 )
      v17(&v16, &v16, 3, v14, v13);
  }
  return v1;
}
