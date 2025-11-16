// Function: sub_2E8DE70
// Address: 0x2e8de70
//
void __fastcall sub_2E8DE70(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  int v3; // ebx
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r14
  __int16 v9; // ax
  __int64 *v10; // rdi
  __int64 *v11; // r13
  __int64 v12; // rcx
  _BYTE *v13; // rsi
  _BYTE *v14; // rdx
  __int64 (__fastcall **v15)(__m128i *, __m128i *, int); // r8
  __int64 (__fastcall **v16)(__m128i *, __m128i *, int); // rdi
  unsigned __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // r9
  __int64 *v23; // [rsp+18h] [rbp-128h]
  int v24; // [rsp+24h] [rbp-11Ch]
  __int64 v25; // [rsp+28h] [rbp-118h]
  __int64 *v26; // [rsp+30h] [rbp-110h] BYREF
  __int64 v27; // [rsp+38h] [rbp-108h]
  _BYTE v28[16]; // [rsp+40h] [rbp-100h] BYREF
  __m128i v29; // [rsp+50h] [rbp-F0h] BYREF
  __int64 (__fastcall *v30)(__m128i *, __m128i *, int); // [rsp+60h] [rbp-E0h] BYREF
  bool (__fastcall *v31)(_DWORD *, __int64); // [rsp+68h] [rbp-D8h]
  void (__fastcall *v32)(_QWORD, _QWORD, _QWORD); // [rsp+70h] [rbp-D0h]
  unsigned __int8 (__fastcall *v33)(_QWORD, _QWORD); // [rsp+78h] [rbp-C8h]
  _QWORD v34[2]; // [rsp+80h] [rbp-C0h] BYREF
  _QWORD v35[2]; // [rsp+90h] [rbp-B0h] BYREF
  void (__fastcall *v36)(_QWORD *, _QWORD *, __int64); // [rsp+A0h] [rbp-A0h]
  __int64 v37; // [rsp+A8h] [rbp-98h]
  __m128i v38; // [rsp+B0h] [rbp-90h] BYREF
  _BYTE v39[16]; // [rsp+C0h] [rbp-80h] BYREF
  void (__fastcall *v40)(__int64 (__fastcall **)(_QWORD *, _DWORD *, int), _BYTE *, __int64); // [rsp+D0h] [rbp-70h]
  unsigned __int8 (__fastcall *v41)(_QWORD, _QWORD); // [rsp+D8h] [rbp-68h]
  __int64 v42; // [rsp+E0h] [rbp-60h]
  __int64 v43; // [rsp+E8h] [rbp-58h]
  _BYTE v44[16]; // [rsp+F0h] [rbp-50h] BYREF
  void (__fastcall *v45)(_QWORD *, _BYTE *, __int64); // [rsp+100h] [rbp-40h]
  __int64 v46; // [rsp+108h] [rbp-38h]

  v26 = (__int64 *)v28;
  v27 = 0x200000000LL;
  v2 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)v2 )
    return;
  v3 = *(_DWORD *)(v2 + 8);
  v24 = v3;
  v5 = sub_2E866D0(a1);
  if ( v3 < 0 )
    v6 = *(_QWORD *)(*(_QWORD *)(v5 + 56) + 16LL * (v3 & 0x7FFFFFFF) + 8);
  else
    v6 = *(_QWORD *)(*(_QWORD *)(v5 + 304) + 8LL * (unsigned int)v3);
  if ( v6 )
  {
    if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
    {
LABEL_7:
      v7 = (unsigned int)v27;
LABEL_8:
      v8 = *(_QWORD *)(v6 + 16);
      v9 = *(_WORD *)(v8 + 68);
      if ( v9 == 14 )
      {
        v21 = *(_QWORD *)(v8 + 32);
        v20 = v21 + 40;
      }
      else
      {
        if ( v9 != 15 )
          goto LABEL_12;
        v19 = *(_QWORD *)(v8 + 32);
        v20 = v19 + 40LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
        v21 = v19 + 80;
      }
      if ( v20 != sub_2E85500(v21, v20, v24) )
      {
        if ( v7 + 1 > (unsigned __int64)HIDWORD(v27) )
        {
          sub_C8D5F0((__int64)&v26, v28, v7 + 1, 8u, v7, v22);
          v7 = (unsigned int)v27;
        }
        v26[v7] = v8;
        v7 = (unsigned int)(v27 + 1);
        LODWORD(v27) = v27 + 1;
      }
LABEL_12:
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 32);
        if ( !v6 )
          goto LABEL_13;
        if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
          goto LABEL_8;
      }
    }
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 32);
      if ( !v6 )
        break;
      if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
        goto LABEL_7;
    }
  }
  v7 = (unsigned int)v27;
LABEL_13:
  v10 = v26;
  v23 = &v26[v7];
  if ( v23 != v26 )
  {
    v11 = v26;
    do
    {
      v12 = *v11;
      v35[0] = 0;
      v29.m128i_i32[0] = v24;
      v25 = v12;
      v31 = sub_2E85490;
      v30 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_2E854D0;
      sub_2E854D0(v34, &v29, 2);
      v35[1] = v31;
      v35[0] = v30;
      v13 = *(_BYTE **)(v25 + 32);
      if ( *(_WORD *)(v25 + 68) == 14 )
      {
        v14 = v13 + 40;
      }
      else
      {
        v14 = &v13[40 * (*(_DWORD *)(v25 + 40) & 0xFFFFFF)];
        v13 += 80;
      }
      sub_2E85EC0(&v38, v13, v14, (__int64)v34);
      if ( v35[0] )
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v35[0])(v34, v34, 3);
      if ( v30 )
        v30(&v29, &v29, 3);
      v32 = 0;
      v29 = v38;
      if ( v40 )
      {
        v40((__int64 (__fastcall **)(_QWORD *, _DWORD *, int))&v30, v39, 2);
        v33 = v41;
        v32 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v40;
      }
      v36 = 0;
      v34[0] = v42;
      v34[1] = v43;
      if ( v45 )
      {
        v45(v35, v44, 2);
        v37 = v46;
        v36 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v45;
      }
      while ( 1 )
      {
        v15 = (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v29.m128i_i64[0];
        if ( v29.m128i_i64[0] == v34[0] )
          break;
        while ( 1 )
        {
          v16 = (__int64 (__fastcall **)(__m128i *, __m128i *, int))v15;
          v17 = a2;
          sub_2EAB0C0((__int64 (__fastcall **)(__m128i *, __m128i *, int))v15, a2);
          v15 = (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))(v29.m128i_i64[0] + 40);
          v29.m128i_i64[0] = (__int64)v15;
          if ( v15 != (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v29.m128i_i64[1] )
            break;
LABEL_30:
          if ( (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v34[0] == v15 )
            goto LABEL_31;
        }
        while ( 1 )
        {
          if ( !v32 )
            sub_4263D6(v16, v17, v18);
          v17 = (unsigned __int64)v15;
          v16 = &v30;
          if ( v33(&v30, (__int64 (__fastcall **)(__m128i *, __m128i *, int))v15) )
            break;
          v15 = (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))(v29.m128i_i64[0] + 40);
          v29.m128i_i64[0] = (__int64)v15;
          if ( (__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v29.m128i_i64[1] == v15 )
            goto LABEL_30;
        }
      }
LABEL_31:
      if ( v36 )
        v36(v35, v35, 3);
      if ( v32 )
        v32(&v30, &v30, 3);
      if ( v45 )
        v45(v44, v44, 3);
      if ( v40 )
        v40((__int64 (__fastcall **)(_QWORD *, _DWORD *, int))v39, v39, 3);
      ++v11;
    }
    while ( v23 != v11 );
    v10 = v26;
  }
  if ( v10 != (__int64 *)v28 )
    _libc_free((unsigned __int64)v10);
}
