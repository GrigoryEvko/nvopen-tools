// Function: sub_1234D00
// Address: 0x1234d00
//
__int64 __fastcall sub_1234D00(__int64 a1, _QWORD *a2, __int64 a3, __int64 *a4)
{
  unsigned __int64 v5; // r15
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 *v8; // rax
  __m128i *v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rax
  __m128i *v13; // rax
  __int64 v14; // rcx
  __m128i *v15; // rax
  __int64 *v16; // r14
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  __int64 *v19; // r14
  _QWORD *v20; // rax
  __int64 v21; // [rsp+0h] [rbp-120h]
  __int64 v22; // [rsp+0h] [rbp-120h]
  __int64 *v23; // [rsp+10h] [rbp-110h] BYREF
  __int64 v24; // [rsp+18h] [rbp-108h] BYREF
  __int64 v25[2]; // [rsp+20h] [rbp-100h] BYREF
  __int64 v26; // [rsp+30h] [rbp-F0h] BYREF
  __m128i *v27; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+48h] [rbp-D8h]
  __m128i v29; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v30[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+70h] [rbp-B0h] BYREF
  __m128i *v32; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v33; // [rsp+88h] [rbp-98h]
  __m128i v34; // [rsp+90h] [rbp-90h] BYREF
  __m128i *v35; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v36; // [rsp+A8h] [rbp-78h]
  __m128i v37; // [rsp+B0h] [rbp-70h] BYREF
  int v38[8]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v39; // [rsp+E0h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 232);
  v23 = 0;
  *(_QWORD *)v38 = "expected type";
  v39 = 259;
  v6 = sub_12190A0(a1, &v23, v38, 1);
  if ( !(_BYTE)v6 )
  {
    v7 = v6;
    v8 = *(__int64 **)(*(_QWORD *)(a4[1] + 24) + 16LL);
    if ( *((_BYTE *)v23 + 8) == 7 )
    {
      if ( *(_BYTE *)(*v8 + 8) != 7 )
      {
        sub_1207630(v25, *v8);
        v9 = (__m128i *)sub_2241130(v25, 0, 0, "value doesn't match function result type '", 42);
        v27 = &v29;
        if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
        {
          v29 = _mm_loadu_si128(v9 + 1);
        }
        else
        {
          v27 = (__m128i *)v9->m128i_i64[0];
          v29.m128i_i64[0] = v9[1].m128i_i64[0];
        }
        v10 = v9->m128i_i64[1];
        v9[1].m128i_i8[0] = 0;
        v28 = v10;
        v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
        v9->m128i_i64[1] = 0;
        if ( v28 != 0x3FFFFFFFFFFFFFFFLL )
        {
          v11 = (__m128i *)sub_2241490(&v27, "'", 1, v10);
          v35 = &v37;
          if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
          {
            v37 = _mm_loadu_si128(v11 + 1);
          }
          else
          {
            v35 = (__m128i *)v11->m128i_i64[0];
            v37.m128i_i64[0] = v11[1].m128i_i64[0];
          }
          v36 = v11->m128i_i64[1];
          v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
          v11->m128i_i64[1] = 0;
          v11[1].m128i_i8[0] = 0;
          v39 = 260;
          *(_QWORD *)v38 = &v35;
          sub_11FD800(a1 + 176, v5, (__int64)v38, 1);
          if ( v35 != &v37 )
            j_j___libc_free_0(v35, v37.m128i_i64[0] + 1);
          if ( v27 != &v29 )
            j_j___libc_free_0(v27, v29.m128i_i64[0] + 1);
          if ( (__int64 *)v25[0] != &v26 )
            j_j___libc_free_0(v25[0], v26 + 1);
          return 1;
        }
LABEL_39:
        sub_4262D8((__int64)"basic_string::append");
      }
      v19 = *(__int64 **)a1;
      v20 = sub_BD2C40(72, 0);
      v18 = v20;
      if ( v20 )
        sub_B4BB80((__int64)v20, (__int64)v19, 0, 0, 0, 0);
      goto LABEL_33;
    }
    v21 = *v8;
    v7 = sub_1224B80((__int64 **)a1, (__int64)v23, &v24, a4);
    if ( !(_BYTE)v7 )
    {
      if ( v21 != *(_QWORD *)(v24 + 8) )
      {
        sub_1207630(v30, v21);
        v13 = (__m128i *)sub_2241130(v30, 0, 0, "value doesn't match function result type '", 42);
        v32 = &v34;
        if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
        {
          v34 = _mm_loadu_si128(v13 + 1);
        }
        else
        {
          v32 = (__m128i *)v13->m128i_i64[0];
          v34.m128i_i64[0] = v13[1].m128i_i64[0];
        }
        v14 = v13->m128i_i64[1];
        v13[1].m128i_i8[0] = 0;
        v33 = v14;
        v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
        v13->m128i_i64[1] = 0;
        if ( v33 != 0x3FFFFFFFFFFFFFFFLL )
        {
          v15 = (__m128i *)sub_2241490(&v32, "'", 1, v14);
          v35 = &v37;
          if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
          {
            v37 = _mm_loadu_si128(v15 + 1);
          }
          else
          {
            v35 = (__m128i *)v15->m128i_i64[0];
            v37.m128i_i64[0] = v15[1].m128i_i64[0];
          }
          v36 = v15->m128i_i64[1];
          v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
          v15->m128i_i64[1] = 0;
          v15[1].m128i_i8[0] = 0;
          v39 = 260;
          *(_QWORD *)v38 = &v35;
          sub_11FD800(a1 + 176, v5, (__int64)v38, 1);
          if ( v35 != &v37 )
            j_j___libc_free_0(v35, v37.m128i_i64[0] + 1);
          if ( v32 != &v34 )
            j_j___libc_free_0(v32, v34.m128i_i64[0] + 1);
          if ( (__int64 *)v30[0] != &v31 )
            j_j___libc_free_0(v30[0], v31 + 1);
          return 1;
        }
        goto LABEL_39;
      }
      v16 = *(__int64 **)a1;
      v22 = v24;
      v17 = sub_BD2C40(72, 1u);
      v18 = v17;
      if ( v17 )
        sub_B4BB80((__int64)v17, (__int64)v16, v22, 1u, 0, 0);
LABEL_33:
      *a2 = v18;
      return v7;
    }
  }
  return 1;
}
