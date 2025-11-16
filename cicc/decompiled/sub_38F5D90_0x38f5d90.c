// Function: sub_38F5D90
// Address: 0x38f5d90
//
__int64 __fastcall sub_38F5D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128 a5, __m128 a6, double a7)
{
  unsigned __int64 v8; // r15
  unsigned int v9; // r14d
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // r8
  unsigned int v18; // r9d
  void (__fastcall *v19)(__int64, __int64, _QWORD); // rcx
  int v20; // eax
  __int64 v21; // [rsp+0h] [rbp-C0h]
  unsigned int v22; // [rsp+Ch] [rbp-B4h]
  void (__fastcall *v23)(__int64, __int64, _QWORD); // [rsp+10h] [rbp-B0h]
  unsigned int v24; // [rsp+18h] [rbp-A8h]
  _QWORD v25[3]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v26; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v27; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-78h]
  char *v29; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v30; // [rsp+58h] [rbp-68h]
  __int16 v31; // [rsp+60h] [rbp-60h]
  char **v32; // [rsp+70h] [rbp-50h] BYREF
  char *v33; // [rsp+78h] [rbp-48h]
  __int16 v34; // [rsp+80h] [rbp-40h]

  v25[0] = a2;
  v25[1] = a3;
  v8 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v9 = sub_38EB9C0(a1, &v26);
  if ( (_BYTE)v9 )
    return 1;
  if ( v26 < 0 )
  {
    v29 = "'";
    v30 = v25;
    v31 = 1283;
    v32 = &v29;
    v34 = 770;
    v33 = "' directive with negative repeat count has no effect";
    sub_38E4170((_QWORD *)a1, v8, (__int64)&v32, 0, 0);
    return v9;
  }
  v31 = 1283;
  v33 = "' directive";
  v34 = 770;
  v29 = "unexpected token in '";
  v30 = v25;
  v32 = &v29;
  if ( (unsigned __int8)sub_3909E20(a1, 25, &v32) )
  {
    return 1;
  }
  else
  {
    v28 = 1;
    v27 = 0;
    if ( (unsigned __int8)sub_38F4CC0(a1, a4, (__int64)&v27, a5, a6, a7, v10, v11, v12)
      || (v33 = "' directive",
          v29 = "unexpected token in '",
          v34 = 770,
          v30 = v25,
          v31 = 1283,
          v32 = &v29,
          v9 = sub_3909E20(a1, 9, &v32),
          (_BYTE)v9) )
    {
      v9 = 1;
    }
    else
    {
      v14 = v26;
      v15 = 0;
      if ( v26 )
      {
        do
        {
          v17 = *(_QWORD *)(a1 + 328);
          v24 = v28;
          v18 = v28 >> 3;
          v19 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 424LL);
          if ( v28 <= 0x40 )
          {
            v16 = (__int64)v27;
          }
          else
          {
            v21 = *(_QWORD *)(a1 + 328);
            v22 = v28 >> 3;
            v23 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 424LL);
            v20 = sub_16A57B0((__int64)&v27);
            v19 = v23;
            v16 = -1;
            v18 = v22;
            v17 = v21;
            if ( v24 - v20 <= 0x40 )
              v16 = *v27;
          }
          ++v15;
          v19(v17, v16, v18);
        }
        while ( v14 != v15 );
      }
    }
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0((unsigned __int64)v27);
  }
  return v9;
}
