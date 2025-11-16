// Function: sub_193DA60
// Address: 0x193da60
//
__int64 __fastcall sub_193DA60(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 *v22; // rcx
  __int64 v23; // r9
  __int64 v24; // r8
  int v25; // ecx
  int v26; // r10d
  __int64 v27; // [rsp+8h] [rbp-158h] BYREF
  __int64 v28; // [rsp+18h] [rbp-148h] BYREF
  __m128i v29; // [rsp+20h] [rbp-140h] BYREF
  __int64 (__fastcall *v30)(const __m128i **, const __m128i *, int); // [rsp+30h] [rbp-130h]
  char (__fastcall *v31)(_QWORD **, __int64 *); // [rsp+38h] [rbp-128h]
  __int64 v32; // [rsp+40h] [rbp-120h]
  __int64 v33; // [rsp+48h] [rbp-118h]
  __int64 v34; // [rsp+50h] [rbp-110h]
  __int64 v35; // [rsp+58h] [rbp-108h]
  const __m128i *v36[2]; // [rsp+60h] [rbp-100h] BYREF
  __int64 (__fastcall *v37)(const __m128i **, const __m128i *, int); // [rsp+70h] [rbp-F0h]
  char (__fastcall *v38)(_QWORD **, __int64 *); // [rsp+78h] [rbp-E8h]
  _BYTE *v39; // [rsp+80h] [rbp-E0h]
  __int64 v40; // [rsp+88h] [rbp-D8h]
  _BYTE v41[128]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+110h] [rbp-50h]
  __int64 v43; // [rsp+118h] [rbp-48h]
  __int64 v44; // [rsp+120h] [rbp-40h]
  __int64 v45; // [rsp+128h] [rbp-38h]

  v2 = 0;
  v27 = a2;
  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return v2;
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F9E06C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_35;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F9E06C);
  v8 = *(__int64 **)(a1 + 8);
  v9 = v7;
  v10 = v7 + 160;
  v11 = *v8;
  v12 = v8[1];
  if ( v11 == v12 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9920C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_36;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9920C)
      + 160;
  v14 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F99CCC, 1u);
  if ( v14 && (v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4F99CCC)) != 0 )
    v16 = v15 + 160;
  else
    v16 = 0;
  v17 = sub_13FC470(v27);
  v28 = v17;
  if ( !v17 )
  {
    v17 = **(_QWORD **)(v27 + 32);
    v28 = v17;
  }
  v29.m128i_i64[0] = (__int64)&v28;
  v29.m128i_i64[1] = (__int64)&v27;
  v18 = *(unsigned int *)(v9 + 208);
  v31 = sub_1939B70;
  v30 = sub_19392D0;
  v19 = 0;
  if ( (_DWORD)v18 )
  {
    v20 = *(_QWORD *)(v9 + 192);
    v21 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v22 = (__int64 *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( v17 == *v22 )
    {
LABEL_18:
      if ( v22 != (__int64 *)(v20 + 16 * v18) )
      {
        v19 = v22[1];
        goto LABEL_20;
      }
    }
    else
    {
      v25 = 1;
      while ( v23 != -8 )
      {
        v26 = v25 + 1;
        v21 = (v18 - 1) & (v25 + v21);
        v22 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v22;
        if ( *v22 == v17 )
          goto LABEL_18;
        v25 = v26;
      }
    }
    v19 = 0;
  }
LABEL_20:
  v32 = v10;
  v34 = v13;
  v35 = v19;
  v33 = v16;
  v37 = 0;
  sub_19392D0(v36, &v29, 2);
  v39 = v41;
  v42 = 0;
  v38 = v31;
  v43 = 0;
  v37 = v30;
  v40 = 0x1000000000LL;
  v44 = 0;
  v45 = 0;
  v2 = sub_193C710(v24);
  j___libc_free_0(v43);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  if ( v37 )
    v37(v36, (const __m128i *)v36, 3);
  if ( v30 )
    v30((const __m128i **)&v29, &v29, 3);
  return v2;
}
