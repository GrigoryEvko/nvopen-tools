// Function: sub_D87B80
// Address: 0xd87b80
//
void __fastcall sub_D87B80(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax
  __m128i *v6; // rax
  __m128i *v7; // rcx
  __m128i *v8; // rdx
  __m128i *v9; // rax
  __m128i *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // [rsp-38h] [rbp-38h] BYREF
  __int64 v13; // [rsp-30h] [rbp-30h]
  _QWORD *v14; // [rsp-28h] [rbp-28h]

  if ( a1 != a2 )
  {
    v3 = a1[2];
    v4 = a1[4];
    v14 = a1;
    v12 = v3;
    v13 = v4;
    if ( v3 )
    {
      *(_QWORD *)(v3 + 8) = 0;
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 )
        v13 = v5;
    }
    else
    {
      v13 = 0;
    }
    a1[2] = 0;
    a1[3] = a1 + 1;
    a1[4] = a1 + 1;
    a1[5] = 0;
    if ( a2[2] )
    {
      v6 = sub_D87780(a2[2], (__int64)(a1 + 1), &v12);
      v7 = v6;
      do
      {
        v8 = v6;
        v6 = (__m128i *)v6[1].m128i_i64[0];
      }
      while ( v6 );
      a1[3] = v8;
      v9 = v7;
      do
      {
        v10 = v9;
        v9 = (__m128i *)v9[1].m128i_i64[1];
      }
      while ( v9 );
      a1[4] = v10;
      v11 = a2[5];
      a1[2] = v7;
      v3 = v12;
      a1[5] = v11;
    }
    sub_D85C20(v3);
  }
}
