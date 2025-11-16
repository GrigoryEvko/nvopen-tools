// Function: sub_2A3AE90
// Address: 0x2a3ae90
//
__int64 *__fastcall sub_2A3AE90(__int64 a1, unsigned int a2)
{
  __int64 *v3; // r12
  __int64 v4; // r14
  unsigned __int64 v5; // r13
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 *result; // rax
  unsigned __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // r13
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  __int64 *j; // [rsp+10h] [rbp-B0h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  __int64 *v39; // [rsp+18h] [rbp-A8h]
  __int64 *i; // [rsp+20h] [rbp-A0h]
  __int64 *v41; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v42; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v43; // [rsp+40h] [rbp-80h] BYREF
  __int64 v44; // [rsp+48h] [rbp-78h]
  __int64 v45; // [rsp+50h] [rbp-70h] BYREF
  __int64 v46; // [rsp+58h] [rbp-68h]

  v3 = *(__int64 **)(a1 + 72);
  v38 = a2;
  for ( i = &v3[*(unsigned int *)(a1 + 80)]; i != v3; ++v3 )
  {
    v4 = *v3;
    v5 = 0;
    v45 = 4098;
    v43 = &v45;
    v46 = v38;
    v44 = 0x800000002LL;
    while ( 1 )
    {
      v6 = *(_QWORD *)(*(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)) + 24LL);
      v7 = 1;
      if ( *(_BYTE *)v6 == 4 )
        v7 = *(unsigned int *)(v6 + 144);
      if ( v5 >= v7 )
        break;
      if ( *(_QWORD *)a1 == sub_B58EB0(v4, v5) )
      {
        v8 = sub_B0DBA0(
               *(_QWORD **)(*(_QWORD *)(v4 + 32 * (2LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) + 24LL),
               v43,
               (unsigned int)v44,
               v5,
               0);
        v9 = *(_QWORD *)(v8 + 8);
        v10 = (__int64 *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v9 & 4) != 0 )
          v10 = (__int64 *)*v10;
        v11 = sub_B9F6F0(v10, (_BYTE *)v8);
        v12 = v4 + 32 * (2LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
        if ( *(_QWORD *)v12 )
        {
          v13 = *(_QWORD *)(v12 + 8);
          **(_QWORD **)(v12 + 16) = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
        }
        *(_QWORD *)v12 = v11;
        if ( v11 )
        {
          v14 = *(_QWORD *)(v11 + 16);
          *(_QWORD *)(v12 + 8) = v14;
          if ( v14 )
            *(_QWORD *)(v14 + 16) = v12 + 8;
          *(_QWORD *)(v12 + 16) = v11 + 16;
          *(_QWORD *)(v11 + 16) = v12;
        }
      }
      ++v5;
    }
    v15 = *(_QWORD *)(v4 - 32);
    if ( !v15 || *(_BYTE *)v15 || *(_QWORD *)(v15 + 24) != *(_QWORD *)(v4 + 80) )
      BUG();
    if ( *(_DWORD *)(v15 + 36) == 68 && *(_QWORD *)a1 == sub_B595C0(v4) )
    {
      v25 = sub_B0D8A0(
              *(_QWORD **)(*(_QWORD *)(v4 + 32 * (5LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) + 24LL),
              (__int64)&v43,
              0,
              0);
      v26 = *(_QWORD *)(v25 + 8);
      v27 = (__int64 *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v26 & 4) != 0 )
        v27 = (__int64 *)*v27;
      v28 = sub_B9F6F0(v27, (_BYTE *)v25);
      v29 = v4 + 32 * (5LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
      if ( *(_QWORD *)v29 )
      {
        v30 = *(_QWORD *)(v29 + 8);
        **(_QWORD **)(v29 + 16) = v30;
        if ( v30 )
          *(_QWORD *)(v30 + 16) = *(_QWORD *)(v29 + 16);
      }
      *(_QWORD *)v29 = v28;
      if ( v28 )
      {
        v31 = *(_QWORD *)(v28 + 16);
        *(_QWORD *)(v29 + 8) = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = v29 + 8;
        *(_QWORD *)(v29 + 16) = v28 + 16;
        *(_QWORD *)(v28 + 16) = v29;
      }
    }
    if ( v43 != &v45 )
      _libc_free((unsigned __int64)v43);
  }
  v16 = *(__int64 **)(a1 + 104);
  result = &v16[*(unsigned int *)(a1 + 112)];
  v41 = v16;
  for ( j = result; j != v41; result = v41 )
  {
    v18 = 0;
    v19 = *v41;
    v45 = 4098;
    v43 = &v45;
    v46 = a2;
    v44 = 0x800000002LL;
    while ( (unsigned int)sub_B12A30(v19) > v18 )
    {
      if ( *(_QWORD *)a1 == sub_B12A50(v19, v18) )
      {
        v20 = (unsigned int)v44;
        v39 = v43;
        v21 = (_QWORD *)sub_B11F60(v19 + 80);
        v22 = sub_B0DBA0(v21, v39, v20, v18, 0);
        sub_B11F20(&v42, v22);
        v23 = *(_QWORD *)(v19 + 80);
        if ( v23 )
          sub_B91220(v19 + 80, v23);
        v24 = v42;
        *(_QWORD *)(v19 + 80) = v42;
        if ( v24 )
          sub_B976B0((__int64)&v42, v24, v19 + 80);
      }
      ++v18;
    }
    if ( *(_BYTE *)(v19 + 64) == 2 && *(unsigned __int8 **)a1 == sub_B13320(v19) )
    {
      v32 = (_QWORD *)sub_B11F60(v19 + 88);
      v33 = sub_B0D8A0(v32, (__int64)&v43, 0, 0);
      sub_B11F20(&v42, v33);
      v34 = *(_QWORD *)(v19 + 88);
      if ( v34 )
        sub_B91220(v19 + 88, v34);
      v35 = v42;
      *(_QWORD *)(v19 + 88) = v42;
      if ( v35 )
        sub_B976B0((__int64)&v42, v35, v19 + 88);
    }
    if ( v43 != &v45 )
      _libc_free((unsigned __int64)v43);
    ++v41;
  }
  return result;
}
