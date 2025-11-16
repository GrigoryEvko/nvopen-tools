// Function: sub_21F89A0
// Address: 0x21f89a0
//
unsigned __int64 __fastcall sub_21F89A0(_QWORD *a1, __int64 a2, unsigned int a3, unsigned __int32 a4)
{
  __int64 v5; // r13
  int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // r8
  int v11; // ecx
  unsigned __int32 v12; // r15d
  __int64 v13; // rdx
  _QWORD *v14; // rdi
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  bool v18; // zf
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // r14
  unsigned __int64 result; // rax
  bool v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rax
  bool v26; // dl
  bool v27; // al
  unsigned __int64 v28; // [rsp-10h] [rbp-90h]
  __int64 v29; // [rsp+0h] [rbp-80h]
  int v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+18h] [rbp-68h] BYREF
  __m128i v33; // [rsp+20h] [rbp-60h] BYREF
  __int64 v34; // [rsp+30h] [rbp-50h]
  __int64 v35; // [rsp+38h] [rbp-48h]
  __int64 v36; // [rsp+40h] [rbp-40h]

  v5 = a3;
  v8 = sub_21F8260((__int64)a1, a2);
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(_QWORD *)(a2 + 64);
  v11 = v8;
  v32 = v10;
  v12 = *(_DWORD *)(v9 + 40 * v5 + 8);
  if ( v10 )
  {
    v30 = v8;
    sub_1623A60((__int64)&v32, v10, 2);
    v9 = *(_QWORD *)(a2 + 32);
    v11 = v30;
  }
  v13 = a1[61];
  v14 = (_QWORD *)a1[62];
  v15 = *(_DWORD *)(*(_QWORD *)(v13 + 280)
                  + 24LL
                  * (*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[60] + 24LL) + 16LL
                                                                                              * (v12 & 0x7FFFFFFF))
                                                     & 0xFFFFFFFFFFFFFFF8LL)
                                         + 24LL)
                   + *(_DWORD *)(v13 + 288)
                   * (unsigned int)((__int64)(*(_QWORD *)(v13 + 264) - *(_QWORD *)(v13 + 256)) >> 3)));
  v16 = *(_QWORD *)(v9 + 40LL * (unsigned int)(v11 + 4) + 24);
  if ( (unsigned int)v16 >= v15 )
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64, __int64 *, _QWORD, _QWORD, _QWORD))(*v14 + 392LL))(
      v14,
      *(_QWORD *)(a2 + 24),
      a2,
      &v32,
      a4,
      v12,
      0);
    result = v28;
  }
  else
  {
    v17 = *(_QWORD *)(v9 + 40LL * (unsigned int)(v11 + 3) + 24);
    if ( v15 == 16 && (_DWORD)v16 == 8 )
    {
      v18 = (_DWORD)v17 == 1;
      v19 = 17664;
      if ( !v18 )
        v19 = 20736;
    }
    else
    {
      v23 = v15 == 32;
      if ( (_DWORD)v16 == 8 && v15 == 32 )
      {
        v18 = (_DWORD)v17 == 1;
        v19 = 17728;
        if ( !v18 )
          v19 = 20800;
      }
      else
      {
        v26 = v15 == 64;
        if ( (_DWORD)v16 == 8 && v26 )
        {
          v18 = (_DWORD)v17 == 1;
          v19 = 17792;
          if ( !v18 )
            v19 = 20864;
        }
        else
        {
          v27 = (_DWORD)v16 == 16;
          if ( v23 && v27 )
          {
            v18 = (_DWORD)v17 == 1;
            v19 = 15616;
            if ( !v18 )
              v19 = 18688;
          }
          else if ( v26 && v27 )
          {
            v18 = (_DWORD)v17 == 1;
            v19 = 15680;
            if ( !v18 )
              v19 = 18752;
          }
          else
          {
            v18 = (_DWORD)v17 == 1;
            v19 = 0x4000;
            if ( !v18 )
              v19 = 19456;
          }
        }
      }
    }
    v20 = v14[1] + v19;
    v29 = *(_QWORD *)(a2 + 24);
    v31 = *(_QWORD *)(v29 + 56);
    if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
    {
      v21 = (__int64)sub_1E0B640(v31, v20, &v32, 0);
      sub_1DD6E10(v29, (__int64 *)a2, v21);
    }
    else
    {
      v21 = (__int64)sub_1E0B640(v31, v20, &v32, 0);
      sub_1DD5BA0((__int64 *)(v29 + 16), v21);
      v24 = *(_QWORD *)a2;
      v25 = *(_QWORD *)v21;
      *(_QWORD *)(v21 + 8) = a2;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v21 = v24 | v25 & 7;
      *(_QWORD *)(v24 + 8) = v21;
      *(_QWORD *)a2 = v21 | *(_QWORD *)a2 & 7LL;
    }
    v33.m128i_i32[2] = a4;
    v33.m128i_i64[0] = 0x10000000;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    sub_1E1A9C0(v21, v31, &v33);
    v33.m128i_i64[0] = 0;
    v34 = 0;
    v33.m128i_i32[2] = v12;
    v35 = 0;
    v36 = 0;
    sub_1E1A9C0(v21, v31, &v33);
    v33.m128i_i64[0] = 1;
    v34 = 0;
    v35 = 0;
    result = sub_1E1A9C0(v21, v31, &v33);
  }
  if ( v32 )
    return sub_161E7C0((__int64)&v32, v32);
  return result;
}
