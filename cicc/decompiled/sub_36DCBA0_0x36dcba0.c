// Function: sub_36DCBA0
// Address: 0x36dcba0
//
__int64 __fastcall sub_36DCBA0(__int64 a1, __int64 a2)
{
  unsigned __int16 v4; // r15
  __int64 v5; // r8
  __int64 v6; // rbx
  __int64 **v7; // r9
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 *v17; // rbx
  unsigned __int16 v18; // ax
  unsigned __int64 v19; // r14
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 *v22; // r15
  __int64 *v23; // r14
  __int64 v24; // rsi
  __int64 *v25; // rdi
  __int64 *v26; // r13
  __int64 v27; // rsi
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int128 v31; // [rsp-10h] [rbp-100h]
  __int64 v32; // [rsp+8h] [rbp-E8h]
  unsigned __int8 v33; // [rsp+37h] [rbp-B9h]
  __int64 *v34; // [rsp+38h] [rbp-B8h]
  __int64 **v35; // [rsp+38h] [rbp-B8h]
  __int64 **v36; // [rsp+38h] [rbp-B8h]
  __m128i v37; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+50h] [rbp-A0h] BYREF
  int v39; // [rsp+58h] [rbp-98h]
  __int64 *v40; // [rsp+60h] [rbp-90h] BYREF
  __int64 v41; // [rsp+68h] [rbp-88h]
  _BYTE v42[32]; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v43; // [rsp+90h] [rbp-60h] BYREF
  __int64 v44; // [rsp+98h] [rbp-58h]
  _BYTE v45[80]; // [rsp+A0h] [rbp-50h] BYREF

  v37 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v4 = *(_WORD *)(*(_QWORD *)(v37.m128i_i64[0] + 48) + 16LL * v37.m128i_u32[2]);
  v33 = sub_307AB50(v4, 0, v37.m128i_i32[0]);
  if ( !v33 )
    return v33;
  v40 = (__int64 *)v42;
  v41 = 0x400000000LL;
  v44 = 0x400000000LL;
  v6 = *(_QWORD *)(v37.m128i_i64[0] + 56);
  v43 = (__int64 *)v45;
  if ( !v6 )
  {
    v33 = 0;
    goto LABEL_31;
  }
  v7 = &v43;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v6 + 16);
      if ( *(_DWORD *)(v8 + 24) == 158 )
      {
        v9 = *(_QWORD *)(v8 + 40);
        if ( *(_QWORD *)v9 == v37.m128i_i64[0] && *(_DWORD *)(v9 + 8) == v37.m128i_i32[2] )
        {
          v10 = *(_QWORD *)(v9 + 40);
          v11 = *(_DWORD *)(v10 + 24);
          if ( v11 == 11 || v11 == 35 )
            break;
        }
      }
LABEL_4:
      v6 = *(_QWORD *)(v6 + 32);
      if ( !v6 )
        goto LABEL_17;
    }
    v12 = *(_QWORD *)(v10 + 96);
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = (_QWORD *)*v13;
    if ( !v13 )
    {
      v29 = (unsigned int)v41;
      v30 = (unsigned int)v41 + 1LL;
      if ( v30 > HIDWORD(v41) )
      {
        v35 = v7;
        sub_C8D5F0((__int64)&v40, v42, v30, 8u, v5, (__int64)v7);
        v29 = (unsigned int)v41;
        v7 = v35;
      }
      v40[v29] = v8;
      LODWORD(v41) = v41 + 1;
      goto LABEL_4;
    }
    if ( v13 != (_QWORD *)1 )
      BUG();
    v14 = (unsigned int)v44;
    v15 = (unsigned int)v44 + 1LL;
    if ( v15 > HIDWORD(v44) )
    {
      v36 = v7;
      sub_C8D5F0((__int64)v7, v45, v15, 8u, v5, (__int64)v7);
      v14 = (unsigned int)v44;
      v7 = v36;
    }
    v43[v14] = v8;
    v6 = *(_QWORD *)(v6 + 32);
    LODWORD(v44) = v44 + 1;
  }
  while ( v6 );
LABEL_17:
  if ( (_DWORD)v41 && (_DWORD)v44 )
  {
    v16 = *(_QWORD *)(a2 + 80);
    v17 = *(__int64 **)(a1 + 64);
    v18 = word_4456580[v4 - 1];
    v38 = v16;
    v19 = v18;
    v20 = v18;
    if ( v16 )
    {
      v32 = v18;
      sub_B96E90((__int64)&v38, v16, 1);
      v20 = v32;
    }
    *((_QWORD *)&v31 + 1) = 1;
    *(_QWORD *)&v31 = &v37;
    v39 = *(_DWORD *)(a2 + 72);
    v21 = sub_33E6B00(v17, 1561, (__int64)&v38, v20, 0, (__int64)&v38, v19, v31);
    if ( v38 )
      sub_B91220((__int64)&v38, v38);
    v22 = v40;
    v23 = &v40[(unsigned int)v41];
    if ( v23 != v40 )
    {
      do
      {
        v24 = *v22++;
        sub_34161C0(*(_QWORD *)(a1 + 64), v24, 0, v21, 0);
        sub_3421DB0(v21);
      }
      while ( v23 != v22 );
    }
    v25 = v43;
    v26 = v43;
    v34 = &v43[(unsigned int)v44];
    if ( v34 != v43 )
    {
      do
      {
        v27 = *v26++;
        sub_34161C0(*(_QWORD *)(a1 + 64), v27, 0, v21, 1u);
        sub_3421DB0(v21);
      }
      while ( v34 != v26 );
      v25 = v43;
    }
  }
  else
  {
    v33 = 0;
    v25 = v43;
  }
  if ( v25 != (__int64 *)v45 )
    _libc_free((unsigned __int64)v25);
LABEL_31:
  if ( v40 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v40);
  return v33;
}
