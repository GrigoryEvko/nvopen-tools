// Function: sub_226A0F0
// Address: 0x226a0f0
//
__int64 __fastcall sub_226A0F0(
        int a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __m128i a7,
        __m128i *a8,
        __int64 a9,
        unsigned __int64 *a10,
        __int64 a11,
        _QWORD *a12)
{
  _DWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r12d
  unsigned __int64 v19; // r8
  __int64 v20; // r14
  __int64 v21; // rbx
  _QWORD *v22; // rdi
  unsigned __int64 v23; // r8
  __int64 v24; // r14
  __int64 v25; // rbx
  _QWORD *v26; // rdi
  _DWORD *v27; // rax
  _DWORD *v28; // rax
  _DWORD *v29; // rax
  unsigned int v31; // [rsp+14h] [rbp-7Ch]
  _QWORD *v32; // [rsp+18h] [rbp-78h] BYREF
  unsigned __int64 v33; // [rsp+20h] [rbp-70h] BYREF
  __int64 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+30h] [rbp-60h]
  unsigned __int64 v36; // [rsp+40h] [rbp-50h] BYREF
  __int64 v37; // [rsp+48h] [rbp-48h]
  __int64 v38; // [rsp+50h] [rbp-40h]

  v32 = a3;
  v14 = (_DWORD *)sub_CEECD0(4, 4u);
  *v14 = 1;
  sub_C94E10((__int64)qword_4F86310, v14);
  v31 = sub_226C400(a1, a2, (_DWORD)v32, a6, a4, *a12, a12[1]);
  if ( *a12 && ((unsigned int (__fastcall *)(_QWORD, _QWORD, __int64, __int64))*a12)(a12[1], 0, v15, v16) )
  {
    return v31;
  }
  else if ( (unsigned __int8)sub_2260560(v32, a6) )
  {
    v35 = 0x1000000000LL;
    v38 = 0x1000000000LL;
    v33 = 0;
    v34 = 0;
    v36 = 0;
    v37 = 0;
    if ( (unsigned __int8)sub_22674E0(
                            a1,
                            a2,
                            (__int64 *)&v32,
                            a4,
                            a5,
                            a6,
                            a7,
                            (__int64)&v33,
                            (__int64)&v36,
                            a8,
                            a9,
                            a10,
                            a11,
                            (__int64)a12) )
    {
      if ( !*a12 || !((unsigned int (__fastcall *)(_QWORD, _QWORD))*a12)(a12[1], 0) )
      {
        v29 = (_DWORD *)sub_CEECD0(4, 4u);
        *v29 = 3;
        sub_C94E10((__int64)qword_4F86310, v29);
        sub_22613E0(v32 + 3, (__int64)&v33, (__int64)&v36);
      }
      v17 = v31;
    }
    else
    {
      v17 = 1;
    }
    v19 = v36;
    if ( HIDWORD(v37) && (_DWORD)v37 )
    {
      v20 = 8LL * (unsigned int)v37;
      v21 = 0;
      do
      {
        v22 = *(_QWORD **)(v19 + v21);
        if ( v22 != (_QWORD *)-8LL && v22 )
        {
          sub_C7D6A0((__int64)v22, *v22 + 17LL, 8);
          v19 = v36;
        }
        v21 += 8;
      }
      while ( v20 != v21 );
    }
    _libc_free(v19);
    if ( HIDWORD(v34) )
    {
      v23 = v33;
      if ( (_DWORD)v34 )
      {
        v24 = 8LL * (unsigned int)v34;
        v25 = 0;
        do
        {
          v26 = *(_QWORD **)(v23 + v25);
          if ( v26 && v26 != (_QWORD *)-8LL )
          {
            sub_C7D6A0((__int64)v26, *v26 + 17LL, 8);
            v23 = v33;
          }
          v25 += 8;
        }
        while ( v24 != v25 );
      }
    }
    else
    {
      v23 = v33;
    }
    _libc_free(v23);
  }
  else
  {
    v27 = (_DWORD *)sub_CEECD0(4, 4u);
    *v27 = 2;
    sub_C94E10((__int64)qword_4F86310, v27);
    v17 = sub_226C400(a1, a2, (_DWORD)v32, a6, a4, *a12, a12[1]);
    v28 = (_DWORD *)sub_CEECD0(4, 4u);
    *v28 = 3;
    sub_C94E10((__int64)qword_4F86310, v28);
  }
  return v17;
}
