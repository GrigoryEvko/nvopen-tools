// Function: sub_259EC90
// Address: 0x259ec90
//
__int64 __fastcall sub_259EC90(__int64 a1, __int64 a2)
{
  unsigned int v4; // r14d
  unsigned __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __m128i v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // edx
  int v15; // ecx
  unsigned int v16; // edx
  __int64 v17; // rdi
  int v18; // r8d
  char v19; // [rsp+Fh] [rbp-51h] BYREF
  __m128i v20; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+28h] [rbp-38h]

  if ( !*(_BYTE *)(a1 + 184) || (v12 = sub_2509740((_QWORD *)(a1 + 72)), sub_259E650(a1, a2, v12)) )
  {
    v4 = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 184) = 0;
    v4 = 0;
  }
  v5 = sub_250D070((_QWORD *)(a1 + 72));
  if ( *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL) != 7 && *(_QWORD *)(v5 + 16) )
  {
    if ( *(_BYTE *)v5 > 0x15u )
    {
      if ( *(_BYTE *)v5 > 0x1Cu )
      {
        v6 = sub_B43CB0(v5);
        v7 = *(_QWORD *)(a2 + 200);
        if ( *(_DWORD *)(v7 + 40) )
        {
          v13 = *(_QWORD *)(v7 + 8);
          v14 = *(_DWORD *)(v7 + 24);
          if ( !v14 )
            goto LABEL_12;
          v15 = v14 - 1;
          v16 = (v14 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v17 = *(_QWORD *)(v13 + 8LL * v16);
          if ( v6 != v17 )
          {
            v18 = 1;
            while ( v17 != -4096 )
            {
              v16 = v15 & (v18 + v16);
              v17 = *(_QWORD *)(v13 + 8LL * v16);
              if ( v6 == v17 )
                goto LABEL_8;
              ++v18;
            }
            goto LABEL_12;
          }
        }
      }
LABEL_8:
      v19 = 0;
      v8.m128i_i64[0] = sub_250D2C0(v5, 0);
      v20 = v8;
      v9 = sub_2527570(a2, &v20, a1, &v19);
      v22 = v10;
      v21 = v9;
      if ( !(_BYTE)v10 || v21 )
        return v4;
    }
    if ( (unsigned __int8)sub_252FFB0(
                            a2,
                            (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2535020,
                            (__int64)&v20,
                            a1,
                            v5,
                            0,
                            0,
                            0,
                            0,
                            0) )
      return v4;
LABEL_12:
    v4 = 0;
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  }
  return v4;
}
