// Function: sub_21F8650
// Address: 0x21f8650
//
unsigned __int64 __fastcall sub_21F8650(_QWORD *a1, __int64 a2, unsigned int a3, unsigned __int32 a4)
{
  __int64 v5; // r13
  int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // r8
  int v11; // ecx
  unsigned __int32 v12; // r14d
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r8
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rsi
  bool v20; // zf
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // r15
  unsigned __int64 result; // rax
  bool v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rax
  bool v28; // al
  bool v29; // dl
  unsigned __int64 v30; // [rsp-10h] [rbp-90h]
  __int64 v31; // [rsp+0h] [rbp-80h]
  int v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+18h] [rbp-68h] BYREF
  __m128i v35; // [rsp+20h] [rbp-60h] BYREF
  __int64 v36; // [rsp+30h] [rbp-50h]
  __int64 v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+40h] [rbp-40h]

  v5 = a3;
  v8 = sub_21F8260((__int64)a1, a2);
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(_QWORD *)(a2 + 64);
  v11 = v8;
  v34 = v10;
  v12 = *(_DWORD *)(v9 + 40 * v5 + 8);
  if ( v10 )
  {
    v32 = v8;
    sub_1623A60((__int64)&v34, v10, 2);
    v9 = *(_QWORD *)(a2 + 32);
    v11 = v32;
  }
  v13 = a1[61];
  v14 = *(_QWORD *)(v9 + 40LL * (unsigned int)(v11 + 4) + 24);
  v15 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[60] + 24LL) + 16LL * (v12 & 0x7FFFFFFF))
                                        & 0xFFFFFFFFFFFFFFF8LL)
                            + 24LL)
      + *(_DWORD *)(v13 + 288) * (unsigned int)((__int64)(*(_QWORD *)(v13 + 264) - *(_QWORD *)(v13 + 256)) >> 3);
  v16 = *(_QWORD *)(v13 + 280);
  v17 = (_QWORD *)a1[62];
  v18 = *(_DWORD *)(v16 + 24 * v15);
  if ( (unsigned int)v14 >= v18 )
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64, __int64 *, _QWORD, _QWORD, _QWORD))(*v17 + 392LL))(
      v17,
      *(_QWORD *)(a2 + 24),
      a2,
      &v34,
      v12,
      a4,
      0);
    result = v30;
  }
  else
  {
    v19 = *(_QWORD *)(v9 + 40LL * (unsigned int)(v11 + 3) + 24);
    if ( v18 == 16 && (_DWORD)v14 == 8 )
    {
      v20 = (_DWORD)v19 == 1;
      v21 = 15744;
      if ( !v20 )
        v21 = 18816;
    }
    else
    {
      v25 = v18 == 32;
      if ( (_DWORD)v14 == 8 && v18 == 32 )
      {
        v20 = (_DWORD)v19 == 1;
        v21 = 16448;
        if ( !v20 )
          v21 = 19520;
      }
      else
      {
        v28 = v18 == 64;
        if ( (_DWORD)v14 == 8 && v28 )
        {
          v20 = (_DWORD)v19 == 1;
          v21 = 17152;
          if ( !v20 )
            v21 = 20224;
        }
        else
        {
          v29 = (_DWORD)v14 == 16;
          if ( v25 && v29 )
          {
            v20 = (_DWORD)v19 == 1;
            v21 = 16256;
            if ( !v20 )
              v21 = 19328;
          }
          else if ( v28 && v29 )
          {
            v20 = (_DWORD)v19 == 1;
            v21 = 16960;
            if ( !v20 )
              v21 = 20032;
          }
          else
          {
            v20 = (_DWORD)v19 == 1;
            v21 = 17024;
            if ( !v20 )
              v21 = 20096;
          }
        }
      }
    }
    v22 = v17[1] + v21;
    v31 = *(_QWORD *)(a2 + 24);
    v33 = *(_QWORD *)(v31 + 56);
    if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
    {
      v23 = (__int64)sub_1E0B640(v33, v22, &v34, 0);
      sub_1DD6E10(v31, (__int64 *)a2, v23);
    }
    else
    {
      v23 = (__int64)sub_1E0B640(v33, v22, &v34, 0);
      sub_1DD5BA0((__int64 *)(v31 + 16), v23);
      v26 = *(_QWORD *)a2;
      v27 = *(_QWORD *)v23;
      *(_QWORD *)(v23 + 8) = a2;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v23 = v26 | v27 & 7;
      *(_QWORD *)(v26 + 8) = v23;
      *(_QWORD *)a2 = v23 | *(_QWORD *)a2 & 7LL;
    }
    v35.m128i_i32[2] = v12;
    v35.m128i_i64[0] = 0x10000000;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    sub_1E1A9C0(v23, v33, &v35);
    v35.m128i_i64[0] = 0;
    v36 = 0;
    v35.m128i_i32[2] = a4;
    v37 = 0;
    v38 = 0;
    sub_1E1A9C0(v23, v33, &v35);
    v35.m128i_i64[0] = 1;
    v36 = 0;
    v37 = 0;
    result = sub_1E1A9C0(v23, v33, &v35);
  }
  if ( v34 )
    return sub_161E7C0((__int64)&v34, v34);
  return result;
}
