// Function: sub_2D92DA0
// Address: 0x2d92da0
//
__int64 __fastcall sub_2D92DA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // rsi
  __int64 v8; // rdx
  int v9; // ebx
  unsigned __int64 *v10; // r12
  unsigned __int64 *v11; // rbx
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  unsigned __int64 *v14; // rbx
  __m128i *v15; // rbx
  __m128i *v16; // r12
  __int64 *v18; // r8
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r15
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned __int64 v28; // rbx
  _QWORD *v29; // rdi
  __m128i *v30; // [rsp+0h] [rbp-70h] BYREF
  __m128i *v31; // [rsp+8h] [rbp-68h]
  __int64 *v32; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 *v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h] BYREF

  sub_F064E0((__int64)&v30, (char *)byte_3F871B3, 0, a4, a5, a6);
  sub_2D90F60((__int64 *)&v32);
  v7 = "native";
  v9 = sub_2241AC0((__int64)&v32, "native");
  if ( v32 != &v34 )
  {
    v7 = (char *)(v34 + 1);
    j_j___libc_free_0((unsigned __int64)v32);
  }
  if ( !v9 )
  {
    sub_1257FE0((__int64)&v32);
    v18 = v32;
    if ( (_DWORD)v33 )
    {
      if ( *v32 != -8 && *v32 )
      {
        v21 = v32;
      }
      else
      {
        v19 = v32 + 1;
        do
        {
          do
          {
            v20 = *v19;
            v21 = v19++;
          }
          while ( v20 == -8 );
        }
        while ( !v20 );
      }
      v22 = &v32[(unsigned int)v33];
      if ( v22 == v21 )
      {
        if ( HIDWORD(v33) )
        {
          v26 = (unsigned int)v33;
          goto LABEL_36;
        }
      }
      else
      {
        do
        {
          v7 = (char *)(*v21 + 16);
          sub_F060D0(&v30, v7, *(_QWORD *)*v21, *(_BYTE *)(*v21 + 8));
          v23 = v21[1];
          if ( v23 && v23 != -8 )
          {
            ++v21;
          }
          else
          {
            v24 = v21 + 2;
            do
            {
              do
              {
                v25 = *v24;
                v21 = v24++;
              }
              while ( v25 == -8 );
            }
            while ( !v25 );
          }
        }
        while ( v21 != v22 );
        v18 = v32;
        if ( HIDWORD(v33) )
        {
          v26 = (unsigned int)v33;
          if ( (_DWORD)v33 )
          {
LABEL_36:
            v27 = 8 * v26;
            v28 = 0;
            do
            {
              v29 = (_QWORD *)v18[v28 / 8];
              if ( v29 && v29 != (_QWORD *)-8LL )
              {
                v7 = (char *)(*v29 + 17LL);
                sub_C7D6A0((__int64)v29, (__int64)v7, 8);
                v18 = v32;
              }
              v28 += 8LL;
            }
            while ( v27 != v28 );
          }
        }
      }
    }
    _libc_free((unsigned __int64)v18);
  }
  sub_2D91020(&v32, (__int64)v7, v8);
  v10 = (unsigned __int64 *)v32;
  v11 = v33;
  if ( v33 != (unsigned __int64 *)v32 )
  {
    do
    {
      v12 = (_BYTE *)*v10;
      v13 = v10[1];
      v10 += 4;
      sub_F060D0(&v30, v12, v13, 1);
    }
    while ( v11 != v10 );
    v14 = v33;
    v10 = (unsigned __int64 *)v32;
    if ( v33 != (unsigned __int64 *)v32 )
    {
      do
      {
        if ( (unsigned __int64 *)*v10 != v10 + 2 )
          j_j___libc_free_0(*v10);
        v10 += 4;
      }
      while ( v14 != v10 );
      v10 = (unsigned __int64 *)v32;
    }
  }
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
  sub_F05F90(a1, (__int64 *)&v30);
  v15 = v31;
  v16 = v30;
  if ( v31 != v30 )
  {
    do
    {
      if ( (__m128i *)v16->m128i_i64[0] != &v16[1] )
        j_j___libc_free_0(v16->m128i_i64[0]);
      v16 += 2;
    }
    while ( v15 != v16 );
    v16 = v30;
  }
  if ( v16 )
    j_j___libc_free_0((unsigned __int64)v16);
  return a1;
}
