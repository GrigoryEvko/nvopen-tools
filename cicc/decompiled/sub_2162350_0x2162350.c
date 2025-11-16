// Function: sub_2162350
// Address: 0x2162350
//
__int64 __fastcall sub_2162350(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int32 a5, __int32 a6, char a7)
{
  __int64 v10; // r15
  __int64 v11; // rsi
  __int16 ***v12; // rcx
  __int16 ***v13; // rsi
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __m128i v23; // [rsp+10h] [rbp-60h] BYREF
  __int64 v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 56);
  v11 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 24LL);
  v12 = (__int16 ***)(*(_QWORD *)(v11 + 16LL * (a5 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  v13 = (__int16 ***)(*(_QWORD *)(v11 + 16LL * (a6 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  v14 = *(_DWORD *)(a1 + 344) * ((__int64)(*(_QWORD *)(a1 + 320) - *(_QWORD *)(a1 + 312)) >> 3);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 336) + 24LL * (v14 + (unsigned int)*((unsigned __int16 *)*v13 + 12))) != *(_DWORD *)(*(_QWORD *)(a1 + 336) + 24LL * ((unsigned int)*((unsigned __int16 *)*v12 + 12) + v14)) )
    sub_16BD130("Copy one register into another with a different width", 1u);
  if ( v12 == &off_4A027A0 )
  {
    v15 = 39424;
  }
  else
  {
    v15 = 39296;
    if ( v12 != &off_4A02720 )
    {
      if ( v12 == &off_4A025A0 )
      {
        v15 = 39552;
        if ( v13 != v12 )
          v15 = 10816;
      }
      else if ( v12 == &off_4A024A0 )
      {
        v15 = 39680;
        if ( v13 != v12 )
          v15 = 11008;
      }
      else
      {
        v15 = 39168;
        if ( v12 != &off_4A02460 )
        {
          if ( v12 == &off_4A02760 )
          {
            v15 = 30528;
            if ( v13 != v12 )
              v15 = 10688;
          }
          else
          {
            v15 = 39552;
            if ( v12 != &off_4A026A0 )
            {
              if ( v12 == &off_4A02620 )
              {
                v15 = 30656;
                if ( v13 != v12 )
                  v15 = 10880;
              }
              else
              {
                v15 = 30784;
                if ( v13 != &off_4A02520 )
                  v15 = 11072;
              }
            }
          }
        }
      }
    }
  }
  v16 = (__int64)sub_1E0B640(v10, *(_QWORD *)(a1 + 8) + v15, a4, 0);
  sub_1DD5BA0((__int64 *)(a2 + 16), v16);
  v17 = *(_QWORD *)v16;
  v18 = *a3;
  *(_QWORD *)(v16 + 8) = a3;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v16 = v18 | v17 & 7;
  *(_QWORD *)(v18 + 8) = v16;
  v19 = *a3;
  v23.m128i_i32[1] = 0;
  v23.m128i_i32[2] = a5;
  v24 = 0;
  *a3 = v16 | v19 & 7;
  v25 = 0;
  v26 = 0;
  v23.m128i_i32[0] = 0x10000000;
  sub_1E1A9C0(v16, v10, &v23);
  v23.m128i_i64[0] = 0;
  v24 = 0;
  *(__int32 *)((char *)v23.m128i_i32 + 3) = (a7 & 1) << 6;
  *(__int32 *)((char *)v23.m128i_i32 + 2) = v23.m128i_i16[1] & 0xF00F;
  v23.m128i_i32[2] = a6;
  v23.m128i_i32[0] &= 0xFFF000FF;
  v25 = 0;
  v26 = 0;
  return sub_1E1A9C0(v16, v10, &v23);
}
