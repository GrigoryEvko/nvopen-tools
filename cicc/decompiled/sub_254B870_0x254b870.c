// Function: sub_254B870
// Address: 0x254b870
//
__m128i *__fastcall sub_254B870(__m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // r14
  __int64 *i; // rbx
  _BYTE *v10; // rax
  __int64 *v11; // rcx
  __int64 *v12; // r14
  _BYTE *v13; // rax
  __int64 *v14; // [rsp+8h] [rbp-98h]
  __m128i *v15; // [rsp+10h] [rbp-90h] BYREF
  __int64 v16; // [rsp+18h] [rbp-88h]
  __m128i v17; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v18[3]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v19; // [rsp+48h] [rbp-58h]
  _BYTE *v20; // [rsp+50h] [rbp-50h]
  __int64 v21; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v22; // [rsp+60h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 97) )
  {
    v22 = (unsigned __int64 *)&v15;
    v21 = 0x100000000LL;
    v15 = &v17;
    v17.m128i_i8[0] = 0;
    v18[0] = &unk_49DD210;
    v16 = 0;
    v18[1] = 0;
    v18[2] = 0;
    v19 = 0;
    v20 = 0;
    sub_CB5980((__int64)v18, 0, 0, 0);
    v3 = sub_904010((__int64)v18, "underlying objects: inter ");
    v4 = sub_CB59D0(v3, *(unsigned int *)(a2 + 256));
    v5 = sub_904010(v4, " objects, intra ");
    v6 = sub_CB59D0(v5, *(unsigned int *)(a2 + 144));
    sub_904010(v6, " objects.\n");
    if ( *(_DWORD *)(a2 + 256) )
    {
      sub_904010((__int64)v18, "inter objects:\n");
      v11 = *(__int64 **)(a2 + 248);
      v14 = &v11[*(unsigned int *)(a2 + 256)];
      if ( v14 != v11 )
      {
        v12 = *(__int64 **)(a2 + 248);
        do
        {
          sub_A69870(*v12, v18, 0);
          v13 = v20;
          if ( (unsigned __int64)v20 < v19 )
          {
            ++v20;
            *v13 = 10;
          }
          else
          {
            sub_CB5D20((__int64)v18, 10);
          }
          ++v12;
        }
        while ( v14 != v12 );
      }
    }
    if ( *(_DWORD *)(a2 + 144) )
    {
      sub_904010((__int64)v18, "intra objects:\n");
      v8 = *(__int64 **)(a2 + 136);
      for ( i = &v8[*(unsigned int *)(a2 + 144)]; i != v8; ++v8 )
      {
        sub_A69870(*v8, v18, 0);
        v10 = v20;
        if ( (unsigned __int64)v20 < v19 )
        {
          ++v20;
          *v10 = 10;
        }
        else
        {
          sub_CB5D20((__int64)v18, 10);
        }
      }
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v15 == &v17 )
    {
      a1[1] = _mm_load_si128(&v17);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v15;
      a1[1].m128i_i64[0] = v17.m128i_i64[0];
    }
    v7 = v16;
    v15 = &v17;
    v16 = 0;
    a1->m128i_i64[1] = v7;
    v17.m128i_i8[0] = 0;
    v18[0] = &unk_49DD210;
    sub_CB5840((__int64)v18);
    sub_2240A30((unsigned __int64 *)&v15);
  }
  else
  {
    sub_253C590(a1->m128i_i64, "<invalid>");
  }
  return a1;
}
