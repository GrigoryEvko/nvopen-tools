// Function: sub_18A4E30
// Address: 0x18a4e30
//
__int64 __fastcall sub_18A4E30(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  _QWORD *v4; // rsi
  __m128i *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // r8
  bool v16; // dl
  unsigned __int8 v17; // [rsp+Fh] [rbp-B1h]
  __int64 v18; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-A8h]
  char v20; // [rsp+20h] [rbp-A0h]
  _QWORD *v21; // [rsp+30h] [rbp-90h] BYREF
  __int16 v22; // [rsp+40h] [rbp-80h]
  _QWORD v23[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v24; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v25[2]; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v26[2]; // [rsp+80h] [rbp-40h] BYREF
  int v27; // [rsp+90h] [rbp-30h]
  _QWORD *v28; // [rsp+98h] [rbp-28h]

  v3 = *a2;
  LOWORD(v26[0]) = 260;
  v4 = v25;
  v25[0] = a1 + 1208;
  sub_393FB40(&v18, v25, v3);
  if ( (v20 & 1) != 0 )
  {
    v4 = v19;
    if ( (_DWORD)v18 )
    {
      (*(void (__fastcall **)(_QWORD *))(*v19 + 32LL))(v25);
      v5 = (__m128i *)sub_2241130(v25, 0, 0, "Could not open profile: ", 24);
      v23[0] = &v24;
      if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
      {
        v24 = _mm_loadu_si128(v5 + 1);
      }
      else
      {
        v23[0] = v5->m128i_i64[0];
        v24.m128i_i64[0] = v5[1].m128i_i64[0];
      }
      v6 = v5->m128i_i64[1];
      v5[1].m128i_i8[0] = 0;
      v23[1] = v6;
      v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
      v5->m128i_i64[1] = 0;
      if ( (_QWORD *)v25[0] != v26 )
        j_j___libc_free_0(v25[0], v26[0] + 1LL);
      v22 = 260;
      v7 = *(_QWORD *)(a1 + 1208);
      v21 = v23;
      v8 = *(_QWORD *)(a1 + 1216);
      v25[1] = 7;
      v26[1] = v8;
      v25[0] = &unk_49ECF18;
      v26[0] = v7;
      v27 = 0;
      v28 = &v21;
      sub_16027F0(v3, (__int64)v25);
      if ( (__m128i *)v23[0] != &v24 )
        j_j___libc_free_0(v23[0], v24.m128i_i64[0] + 1);
      result = 0;
      if ( (v20 & 1) != 0 )
        return result;
LABEL_16:
      if ( v18 )
      {
        v17 = result;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
        return v17;
      }
      return result;
    }
  }
  v10 = v18;
  v11 = *(_QWORD *)(a1 + 1192);
  v18 = 0;
  *(_QWORD *)(a1 + 1192) = v10;
  if ( v11 )
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v11 + 8LL))(v11, v4);
  v12 = sub_393D180(v11, v4);
  v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 1192) + 24LL))(*(_QWORD *)(a1 + 1192));
  v15 = v14;
  v16 = 0;
  if ( v12 == v15 )
    v16 = v13 == 0;
  *(_BYTE *)(a1 + 1240) = v16;
  result = 1;
  if ( (v20 & 1) == 0 )
    goto LABEL_16;
  return result;
}
