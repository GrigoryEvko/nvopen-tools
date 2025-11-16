// Function: sub_3266760
// Address: 0x3266760
//
void __fastcall sub_3266760(__m128i *src, const __m128i *a2)
{
  const __m128i *i; // rbx
  __int64 v4; // rcx
  __int64 v5; // r15
  __int32 v6; // r14d
  __int64 v7; // r13
  char v8; // r14
  unsigned __int64 v9; // r13
  unsigned __int16 *v10; // rdx
  unsigned __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  char *v16; // rax
  unsigned __int64 v17; // r14
  unsigned __int16 *v18; // rdx
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  char v23; // al
  unsigned __int64 v24; // rax
  const __m128i *v25; // rdi
  unsigned __int64 v26; // r14
  unsigned int v27; // eax
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r14
  unsigned int v31; // eax
  __int64 v32; // [rsp+18h] [rbp-78h]
  char v33; // [rsp+18h] [rbp-78h]
  unsigned int v34; // [rsp+18h] [rbp-78h]
  unsigned int v35; // [rsp+18h] [rbp-78h]
  unsigned __int16 v36; // [rsp+20h] [rbp-70h] BYREF
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h]
  __int64 v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h]
  __int64 v41; // [rsp+48h] [rbp-48h]
  unsigned __int64 v42; // [rsp+50h] [rbp-40h] BYREF
  __int64 v43; // [rsp+58h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = src + 2; a2 != i; src[1].m128i_i64[1] = v7 )
    {
      while ( 1 )
      {
        v8 = *(_BYTE *)sub_2E79000(*(__int64 **)(i[1].m128i_i64[1] + 40));
        v9 = (unsigned __int32)i[1].m128i_i32[0] >> 3;
        v10 = *(unsigned __int16 **)(i->m128i_i64[1] + 48);
        v11 = *v10;
        v12 = *((_QWORD *)v10 + 1);
        v36 = v11;
        v37 = v12;
        if ( v11 )
        {
          if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
LABEL_32:
            BUG();
          v14 = 16LL * (v11 - 1);
          v13 = *(_QWORD *)&byte_444C4A0[v14];
          LOBYTE(v14) = byte_444C4A0[v14 + 8];
        }
        else
        {
          v13 = sub_3007260((__int64)&v36);
          v38 = v13;
          v39 = v14;
        }
        v42 = v13;
        LOBYTE(v43) = v14;
        v15 = sub_CA1930(&v42);
        if ( v8 )
        {
          v29 = (unsigned int)(v15 >> 3);
          sub_3266230((__int64)&v42, (__int64)i);
          v30 = v29 - v9;
          if ( (unsigned int)v43 > 0x40 )
          {
            v31 = sub_C44630((__int64)&v42);
            if ( v42 )
            {
              v35 = v31;
              j_j___libc_free_0_0(v42);
              v31 = v35;
            }
          }
          else
          {
            v31 = sub_39FAC40(v42);
          }
          v9 = v30 - (v31 >> 3);
        }
        v16 = (char *)sub_2E79000(*(__int64 **)(src[1].m128i_i64[1] + 40));
        v17 = (unsigned __int32)src[1].m128i_i32[0] >> 3;
        v33 = *v16;
        v18 = *(unsigned __int16 **)(src->m128i_i64[1] + 48);
        v19 = *v18;
        v20 = *((_QWORD *)v18 + 1);
        LOWORD(v42) = v19;
        v43 = v20;
        if ( v19 )
        {
          if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
            goto LABEL_32;
          v28 = 16LL * (v19 - 1);
          v22 = *(_QWORD *)&byte_444C4A0[v28];
          v23 = byte_444C4A0[v28 + 8];
        }
        else
        {
          v40 = sub_3007260((__int64)&v42);
          v41 = v21;
          v22 = v40;
          v23 = v41;
        }
        v42 = v22;
        LOBYTE(v43) = v23;
        v24 = sub_CA1930(&v42);
        if ( v33 )
        {
          v26 = (unsigned int)(v24 >> 3) - v17;
          sub_3266230((__int64)&v42, (__int64)src);
          if ( (unsigned int)v43 > 0x40 )
          {
            v27 = sub_C44630((__int64)&v42);
            if ( v42 )
            {
              v34 = v27;
              j_j___libc_free_0_0(v42);
              v27 = v34;
            }
          }
          else
          {
            v27 = sub_39FAC40(v42);
          }
          v17 = v26 - (v27 >> 3);
        }
        if ( v17 > v9 )
          break;
        v25 = i;
        i += 2;
        sub_32664A0(v25);
        if ( a2 == i )
          return;
      }
      v4 = i->m128i_i64[0];
      v5 = i->m128i_i64[1];
      v6 = i[1].m128i_i32[0];
      v7 = i[1].m128i_i64[1];
      if ( src != i )
      {
        v32 = i->m128i_i64[0];
        memmove(&src[2], src, (char *)i - (char *)src);
        v4 = v32;
      }
      src->m128i_i64[0] = v4;
      i += 2;
      src->m128i_i64[1] = v5;
      src[1].m128i_i32[0] = v6;
    }
  }
}
