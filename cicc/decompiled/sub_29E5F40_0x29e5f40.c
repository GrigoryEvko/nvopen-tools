// Function: sub_29E5F40
// Address: 0x29e5f40
//
void __fastcall sub_29E5F40(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rcx
  _QWORD *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // rbx
  __int64 v15; // r9
  __m128i *v16; // rbx
  _QWORD *v17; // rax
  __m128i *v18; // rbx
  __int64 v19; // rdx
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v21; // r12
  unsigned __int64 v22; // rbx
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __m128i *v26; // rdx
  _QWORD *v27; // rax
  int v28; // r15d
  __int64 *v29; // [rsp+0h] [rbp-E0h]
  __int64 v30; // [rsp+8h] [rbp-D8h]
  __int64 v31; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v32; // [rsp+38h] [rbp-A8h] BYREF
  __m128i *v33; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+48h] [rbp-98h]
  __m128i v35; // [rsp+50h] [rbp-90h] BYREF
  __m128i *v36; // [rsp+60h] [rbp-80h] BYREF
  __int64 v37; // [rsp+68h] [rbp-78h]
  _BYTE v38[112]; // [rsp+70h] [rbp-70h] BYREF

  v30 = a2;
  if ( !a1 )
    BUG();
  v2 = *(_QWORD *)(a1 + 32);
  v31 = a1 + 24;
  if ( v2 != a1 + 24 )
  {
    while ( 1 )
    {
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 8);
      if ( (unsigned __int8)(*(_BYTE *)(v3 - 24) - 34) > 0x33u )
        goto LABEL_14;
      v4 = 0x8000000000041LL;
      if ( !_bittest64(&v4, (unsigned int)*(unsigned __int8 *)(v3 - 24) - 34) )
        goto LABEL_14;
      v5 = (_QWORD *)(v3 - 24);
      if ( *(char *)(v3 - 17) < 0 )
      {
        v6 = sub_BD2BC0(v3 - 24);
        v8 = v6 + v7;
        if ( *(char *)(v3 - 17) < 0 )
          v8 -= sub_BD2BC0(v3 - 24);
        v9 = v8 >> 4;
        if ( (_DWORD)v9 )
        {
          v10 = 0;
          v11 = 16LL * (unsigned int)v9;
          do
          {
            v12 = 0;
            if ( *(char *)(v3 - 17) < 0 )
              v12 = sub_BD2BC0(v3 - 24);
            if ( *(_DWORD *)(*(_QWORD *)(v12 + v10) + 8LL) == 1 )
              goto LABEL_14;
            v10 += 16;
          }
          while ( v10 != v11 );
        }
      }
      v13 = sub_BD3990(*(unsigned __int8 **)(v3 - 56), a2);
      v14 = v13;
      if ( !*v13 && (v13[33] & 0x20) != 0 )
      {
        a2 = 41;
        if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v3 + 48), 41) || (a2 = 41, (unsigned __int8)sub_B49560(v3 - 24, 41)) )
        {
          if ( !sub_B58D90(*((_DWORD *)v14 + 9)) )
            goto LABEL_14;
        }
      }
      v36 = (__m128i *)v38;
      v37 = 0x100000000LL;
      sub_B56970(v3 - 24, (__int64)&v36);
      if ( (unsigned int)v37 >= HIDWORD(v37) )
      {
        v25 = sub_C8D7D0((__int64)&v36, (__int64)v38, 0, 0x38u, &v32, v15);
        v33 = &v35;
        v18 = (__m128i *)v25;
        v35.m128i_i64[0] = 0x74656C636E7566LL;
        v26 = (__m128i *)(v25 + 56LL * (unsigned int)v37);
        v34 = 7;
        if ( v26 )
        {
          v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
          if ( v33 == &v35 )
          {
            v26[1] = _mm_load_si128(&v35);
          }
          else
          {
            v26->m128i_i64[0] = (__int64)v33;
            v26[1].m128i_i64[0] = v35.m128i_i64[0];
          }
          v29 = (__int64 *)v26;
          v26->m128i_i64[1] = v34;
          v33 = &v35;
          v34 = 0;
          v35.m128i_i8[0] = 0;
          v26[2].m128i_i64[0] = 0;
          v26[2].m128i_i64[1] = 0;
          v26[3].m128i_i64[0] = 0;
          v27 = (_QWORD *)sub_22077B0(8u);
          v29[4] = (__int64)v27;
          v29[6] = (__int64)(v27 + 1);
          *v27 = v30;
          v29[5] = (__int64)(v27 + 1);
          if ( v33 != &v35 )
            j_j___libc_free_0((unsigned __int64)v33);
        }
        sub_B56820((__int64)&v36, v18);
        v28 = v32;
        if ( v36 != (__m128i *)v38 )
          _libc_free((unsigned __int64)v36);
        v36 = v18;
        HIDWORD(v37) = v28;
        v19 = (unsigned int)(v37 + 1);
        LODWORD(v37) = v37 + 1;
      }
      else
      {
        v33 = &v35;
        sub_29DF980((__int64 *)&v33, "funclet", (__int64)"");
        v16 = (__m128i *)((char *)v36 + 56 * (unsigned int)v37);
        if ( v16 )
        {
          v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
          if ( v33 == &v35 )
          {
            v16[1] = _mm_load_si128(&v35);
          }
          else
          {
            v16->m128i_i64[0] = (__int64)v33;
            v16[1].m128i_i64[0] = v35.m128i_i64[0];
          }
          v16->m128i_i64[1] = v34;
          v33 = &v35;
          v34 = 0;
          v35.m128i_i8[0] = 0;
          v16[2].m128i_i64[0] = 0;
          v16[2].m128i_i64[1] = 0;
          v16[3].m128i_i64[0] = 0;
          v17 = (_QWORD *)sub_22077B0(8u);
          v16[2].m128i_i64[0] = (__int64)v17;
          v16[3].m128i_i64[0] = (__int64)(v17 + 1);
          *v17 = v30;
          v16[2].m128i_i64[1] = (__int64)(v17 + 1);
        }
        if ( v33 != &v35 )
          j_j___libc_free_0((unsigned __int64)v33);
        v18 = v36;
        v19 = (unsigned int)(v37 + 1);
        LODWORD(v37) = v37 + 1;
      }
      v20 = (unsigned __int8 *)(v3 - 24);
      v21 = (unsigned __int8 *)sub_B4BA60((unsigned __int8 *)(v3 - 24), (__int64)v18, v19, v3, 0);
      sub_BD6B90(v21, v20);
      a2 = (__int64)v21;
      sub_BD84D0((__int64)v5, (__int64)v21);
      sub_B43D60(v5);
      v22 = (unsigned __int64)v36;
      v23 = (unsigned __int64 *)v36 + 7 * (unsigned int)v37;
      if ( v36 != (__m128i *)v23 )
      {
        do
        {
          v24 = *(v23 - 3);
          v23 -= 7;
          if ( v24 )
          {
            a2 = v23[6] - v24;
            j_j___libc_free_0(v24);
          }
          if ( (unsigned __int64 *)*v23 != v23 + 2 )
          {
            a2 = v23[2] + 1;
            j_j___libc_free_0(*v23);
          }
        }
        while ( (unsigned __int64 *)v22 != v23 );
        v23 = (unsigned __int64 *)v36;
      }
      if ( v23 == (unsigned __int64 *)v38 )
      {
LABEL_14:
        if ( v31 == v2 )
          return;
      }
      else
      {
        _libc_free((unsigned __int64)v23);
        if ( v31 == v2 )
          return;
      }
    }
  }
}
