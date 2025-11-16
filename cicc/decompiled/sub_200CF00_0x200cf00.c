// Function: sub_200CF00
// Address: 0x200cf00
//
__int64 __fastcall sub_200CF00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __m128i a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rcx
  _QWORD *v12; // r12
  _QWORD *v13; // rax
  _QWORD *v15; // rdx
  __int64 v16[6]; // [rsp+0h] [rbp-100h] BYREF
  __int64 v17; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD *v18; // [rsp+38h] [rbp-C8h]
  _QWORD *v19; // [rsp+40h] [rbp-C0h]
  __int64 v20; // [rsp+48h] [rbp-B8h]
  int v21; // [rsp+50h] [rbp-B0h]
  _QWORD v22[21]; // [rsp+58h] [rbp-A8h] BYREF

  v18 = v22;
  v19 = v22;
  v8 = *a1;
  v16[2] = (__int64)a1;
  v16[0] = v8;
  v9 = a1[2];
  v16[3] = (__int64)&v17;
  v16[1] = v9;
  v16[4] = a3;
  v20 = 0x100000010LL;
  v21 = 0;
  v22[0] = a2;
  v17 = 1;
  sub_1FFB890(v16, a2, a6, a7, a8, a3, a4, a5);
  v10 = (unsigned __int64)v19;
  v11 = v18;
  if ( v19 == v18 )
  {
    v12 = &v19[HIDWORD(v20)];
    if ( v19 == v12 )
    {
      v15 = v19;
      v13 = v19;
    }
    else
    {
      v13 = v19;
      do
      {
        if ( a2 == *v13 )
          break;
        ++v13;
      }
      while ( v12 != v13 );
      v15 = &v19[HIDWORD(v20)];
    }
  }
  else
  {
    v12 = &v19[(unsigned int)v20];
    v13 = sub_16CC9F0((__int64)&v17, a2);
    if ( a2 == *v13 )
    {
      v10 = (unsigned __int64)v19;
      v11 = v18;
      if ( v19 == v18 )
        v15 = &v19[HIDWORD(v20)];
      else
        v15 = &v19[(unsigned int)v20];
    }
    else
    {
      v10 = (unsigned __int64)v19;
      v11 = v18;
      if ( v19 != v18 )
      {
        LOBYTE(v12) = v12 != &v19[(unsigned int)v20];
LABEL_5:
        _libc_free(v10);
        return (unsigned int)v12;
      }
      v13 = &v19[HIDWORD(v20)];
      v15 = v13;
    }
  }
  while ( v15 != v13 && *v13 >= 0xFFFFFFFFFFFFFFFELL )
    ++v13;
  LOBYTE(v12) = v12 != v13;
  if ( (_QWORD *)v10 != v11 )
    goto LABEL_5;
  return (unsigned int)v12;
}
