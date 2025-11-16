// Function: sub_3940900
// Address: 0x3940900
//
__int64 __fastcall sub_3940900(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i *v9; // rsi
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // [rsp+0h] [rbp-90h] BYREF
  char v13; // [rsp+10h] [rbp-80h]
  __int64 v14; // [rsp+20h] [rbp-70h] BYREF
  char v15; // [rsp+30h] [rbp-60h]
  _QWORD v16[2]; // [rsp+40h] [rbp-50h] BYREF
  char v17; // [rsp+50h] [rbp-40h]

  sub_393FF90((__int64)&v12, a1);
  if ( (v13 & 1) == 0 || (result = (unsigned int)v12, !(_DWORD)v12) )
  {
    sub_393FF90((__int64)&v14, a1);
    if ( (v15 & 1) == 0 || (result = (unsigned int)v14, !(_DWORD)v14) )
    {
      v4 = v16;
      sub_393FF90((__int64)v16, a1);
      if ( (v17 & 1) == 0 || (result = LODWORD(v16[0]), v5 = v16[1], !LODWORD(v16[0])) )
      {
        v9 = (__m128i *)a2[1];
        if ( v9 == (__m128i *)a2[2] )
        {
          v4 = a2;
          sub_3940740(a2, v9, &v12, &v14, v16);
        }
        else
        {
          if ( v9 )
          {
            v5 = v14;
            v10 = v16[0];
            v6 = v12;
            v9->m128i_i64[1] = v14;
            v9->m128i_i32[0] = v6;
            v9[1].m128i_i64[0] = v10;
            v9 = (__m128i *)a2[1];
          }
          v9 = (__m128i *)((char *)v9 + 24);
          a2[1] = (unsigned __int64)v9;
        }
        sub_393D180((__int64)v4, (__int64)v9, v5, v6, v7, v8);
        return 0;
      }
    }
  }
  return result;
}
