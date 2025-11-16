// Function: sub_19DD950
// Address: 0x19dd950
//
__int64 *__fastcall sub_19DD950(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 v10; // rax
  __int64 *result; // rax
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r13
  __int64 v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // r13
  _QWORD *v27; // rdi
  __int64 *v28; // r13
  __int64 v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  __int64 v37; // [rsp+20h] [rbp-60h] BYREF
  __int64 v38; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v39; // [rsp+30h] [rbp-50h] BYREF
  __int64 v40; // [rsp+38h] [rbp-48h]
  __int64 v41; // [rsp+40h] [rbp-40h] BYREF
  __int64 v42; // [rsp+48h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 8);
  v37 = 0;
  v38 = 0;
  if ( !v10 || *(_QWORD *)(v10 + 8) || !(unsigned __int8)sub_19DD690(a1, a4, a2, &v37, &v38) )
    goto LABEL_3;
  v16 = sub_146F1B0(*(_QWORD *)(a1 + 24), v37);
  v33 = sub_146F1B0(*(_QWORD *)(a1 + 24), v38);
  v17 = sub_146F1B0(*(_QWORD *)(a1 + 24), a3);
  v18 = v17;
  if ( v33 == v17
    || (v30 = v17,
        v31 = v38,
        v19 = sub_19DD730(a1, a4, v16, v17, a5, a6),
        result = sub_19DD8D0(a1, v19, v31, a4),
        v18 = v30,
        !result) )
  {
    if ( v16 == v18
      || (v20 = v37, v21 = sub_19DD730(a1, a4, v33, v18, a5, a6), (result = sub_19DD8D0(a1, v21, v20, a4)) == 0) )
    {
LABEL_3:
      if ( *(_BYTE *)(a4 + 16) != 39 )
        return 0;
      v12 = *(_BYTE *)(a2 + 16);
      if ( v12 == 35 )
      {
        v14 = *(_QWORD *)(a2 - 48);
        if ( !v14 )
          return 0;
        v15 = *(_QWORD *)(a2 - 24);
        v37 = *(_QWORD *)(a2 - 48);
        if ( !v15 )
          return 0;
      }
      else
      {
        if ( v12 != 5 )
          return 0;
        if ( *(_WORD *)(a2 + 18) != 11 )
          return 0;
        v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v14 = *(_QWORD *)(a2 - 24 * v13);
        if ( !v14 )
          return 0;
        v37 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v15 = *(_QWORD *)(a2 + 24 * (1 - v13));
        if ( !v15 )
          return 0;
      }
      v22 = *(_QWORD *)(a1 + 24);
      v38 = v15;
      v23 = sub_146F1B0(v22, v14);
      v34 = sub_146F1B0(*(_QWORD *)(a1 + 24), v38);
      v24 = sub_146F1B0(*(_QWORD *)(a1 + 24), a3);
      v25 = *(_QWORD **)(a1 + 24);
      v32 = v24;
      v42 = v24;
      v41 = v23;
      v39 = &v41;
      v40 = 0x200000002LL;
      v26 = sub_147EE30(v25, &v39, 0, 0, a5, a6);
      if ( v39 != &v41 )
        _libc_free((unsigned __int64)v39);
      v27 = *(_QWORD **)(a1 + 24);
      v39 = &v41;
      v41 = v34;
      v42 = v32;
      v40 = 0x200000002LL;
      v35 = sub_147EE30(v27, &v39, 0, 0, a5, a6);
      if ( v39 != &v41 )
        _libc_free((unsigned __int64)v39);
      v28 = (__int64 *)sub_19DD7C0(a1, v26, a4);
      if ( v28 )
      {
        v29 = sub_19DD7C0(a1, v35, a4);
        if ( v29 )
        {
          LOWORD(v41) = 257;
          v36 = sub_15FB440(11, v28, v29, (__int64)&v39, a4);
          sub_164B7C0(v36, a4);
          return (__int64 *)v36;
        }
      }
      return 0;
    }
  }
  return result;
}
