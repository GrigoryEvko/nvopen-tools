// Function: sub_12E7B90
// Address: 0x12e7b90
//
_QWORD *__fastcall sub_12E7B90(
        _QWORD *a1,
        __int64 *a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        _QWORD *a9)
{
  _DWORD *v11; // rax
  _QWORD *v12; // r12
  __int64 v14; // r8
  __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rdi
  _DWORD *v22; // rax
  __int64 *v23; // rax
  _DWORD *v24; // rax
  _QWORD *v27; // [rsp+18h] [rbp-78h] BYREF
  __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h]
  __int64 v31; // [rsp+40h] [rbp-50h] BYREF
  __int64 v32; // [rsp+48h] [rbp-48h]
  __int64 v33; // [rsp+50h] [rbp-40h]

  v27 = a1;
  v11 = (_DWORD *)sub_1C42D70(4, 4);
  *v11 = 1;
  sub_16D40E0(qword_4FBB3B0, v11);
  sub_12E54A0(v27, a2[2], a2[3], a4, a9);
  if ( *a9 && ((unsigned int (__fastcall *)(_QWORD, _QWORD))*a9)(a9[1], 0) )
    return v27;
  v28 = 0;
  v29 = 0;
  v30 = 0x1000000000LL;
  v31 = 0;
  v32 = 0;
  v33 = 0x1000000000LL;
  if ( (unsigned __int8)sub_12D4250(v27, a4) )
  {
    if ( (unsigned __int8)sub_12E1EF0(
                            (__int64 *)&v27,
                            (__int64)a2,
                            a3,
                            a4,
                            (__int64)&v28,
                            (__int64)&v31,
                            a5,
                            a6,
                            a7,
                            a8,
                            (__int64)a9) )
    {
      if ( !*a9 || (a2 = 0, !((unsigned int (__fastcall *)(_QWORD, _QWORD))*a9)(a9[1], 0)) )
      {
        v24 = (_DWORD *)sub_1C42D70(4, 4);
        *v24 = 3;
        sub_16D40E0(qword_4FBB3B0, v24);
        a2 = &v28;
        sub_12D5520(v27 + 3, (__int64)&v28, (__int64)&v31);
      }
    }
    v12 = v27;
  }
  else
  {
    v22 = (_DWORD *)sub_1C42D70(4, 4);
    *v22 = 2;
    sub_16D40E0(qword_4FBB3B0, v22);
    sub_12E54A0(v27, a2[2], a2[3], a4, a9);
    v23 = (__int64 *)sub_1C42D70(4, 4);
    *(_DWORD *)v23 = 3;
    a2 = v23;
    sub_16D40E0(qword_4FBB3B0, v23);
    v12 = v27;
  }
  v14 = v31;
  if ( HIDWORD(v32) && (_DWORD)v32 )
  {
    v15 = 8LL * (unsigned int)v32;
    v16 = 0;
    do
    {
      v17 = *(_QWORD *)(v14 + v16);
      if ( v17 != -8 && v17 )
      {
        _libc_free(v17, a2);
        v14 = v31;
      }
      v16 += 8;
    }
    while ( v15 != v16 );
  }
  _libc_free(v14, a2);
  if ( HIDWORD(v29) )
  {
    v18 = v28;
    if ( (_DWORD)v29 )
    {
      v19 = 8LL * (unsigned int)v29;
      v20 = 0;
      do
      {
        v21 = *(_QWORD *)(v18 + v20);
        if ( v21 && v21 != -8 )
        {
          _libc_free(v21, a2);
          v18 = v28;
        }
        v20 += 8;
      }
      while ( v19 != v20 );
    }
    _libc_free(v18, a2);
  }
  else
  {
    _libc_free(v28, a2);
  }
  return v12;
}
