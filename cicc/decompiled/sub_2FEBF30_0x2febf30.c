// Function: sub_2FEBF30
// Address: 0x2febf30
//
_QWORD *__fastcall sub_2FEBF30(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 *v6; // rdi
  _QWORD *v7; // r12
  int v8; // esi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v13; // r15
  __int16 v14; // bx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i v18[2]; // [rsp+0h] [rbp-180h] BYREF
  char v19; // [rsp+20h] [rbp-160h]
  char v20; // [rsp+21h] [rbp-15Fh]
  __m128i v21[2]; // [rsp+30h] [rbp-150h] BYREF
  char v22; // [rsp+50h] [rbp-130h]
  char v23; // [rsp+51h] [rbp-12Fh]
  __m128i v24[3]; // [rsp+60h] [rbp-120h] BYREF
  __m128i v25[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+B0h] [rbp-D0h]
  __m128i v27[2]; // [rsp+C0h] [rbp-C0h] BYREF
  char v28; // [rsp+E0h] [rbp-A0h]
  char v29; // [rsp+E1h] [rbp-9Fh]
  __m128i v30; // [rsp+F0h] [rbp-90h] BYREF
  char v31; // [rsp+110h] [rbp-70h]
  char v32; // [rsp+111h] [rbp-6Fh]
  __m128i v33[2]; // [rsp+120h] [rbp-60h] BYREF
  char v34; // [rsp+140h] [rbp-40h]
  char v35; // [rsp+141h] [rbp-3Fh]

  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 48) + 72LL) + 40LL);
  v5 = sub_BA8B30(v4, (__int64)"__safestack_unsafe_stack_ptr", 0x1Cu);
  if ( v5 )
  {
    v6 = *(__int64 **)v4;
    v7 = (_QWORD *)v5;
    v8 = *(_DWORD *)(v4 + 316);
    if ( *(_BYTE *)v5 == 3 )
    {
      if ( *(_QWORD *)(v5 + 24) == sub_BCE3C0(v6, v8) )
      {
        if ( a3 == ((*((_BYTE *)v7 + 33) & 0x1C) != 0) )
          return v7;
        v32 = 1;
        v30.m128i_i64[0] = (__int64)"be thread-local";
        v31 = 3;
        if ( a3 )
        {
          v26 = 257;
        }
        else
        {
          v25[0].m128i_i64[0] = (__int64)"not ";
          v26 = 259;
        }
        v23 = 1;
        v21[0].m128i_i64[0] = (__int64)" must ";
        v22 = 3;
        v20 = 1;
        v18[0].m128i_i64[0] = (__int64)"__safestack_unsafe_stack_ptr";
        v19 = 3;
        sub_9C6370(v24, v18, v21, v9, v10, v11);
        sub_9C6370(v27, v24, v25, v15, v16, v17);
      }
      else
      {
        v32 = 1;
        v30.m128i_i64[0] = (__int64)" must have void* type";
        v31 = 3;
        v29 = 1;
        v27[0].m128i_i64[0] = (__int64)"__safestack_unsafe_stack_ptr";
        v28 = 3;
      }
      sub_9C6370(v33, v27, &v30, v9, v10, v11);
      sub_C64D30((__int64)v33, 1u);
    }
    v13 = (_QWORD *)sub_BCE3C0(v6, v8);
  }
  else
  {
    v13 = (_QWORD *)sub_BCE3C0(*(__int64 **)v4, *(_DWORD *)(v4 + 316));
  }
  v35 = 1;
  v33[0].m128i_i64[0] = (__int64)"__safestack_unsafe_stack_ptr";
  v34 = 3;
  v30.m128i_i8[4] = 0;
  v14 = a3 != 0 ? 3 : 0;
  v7 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v7 )
    sub_B30000((__int64)v7, v4, v13, 0, 0, 0, (__int64)v33, 0, v14, v30.m128i_i64[0], 0);
  return v7;
}
