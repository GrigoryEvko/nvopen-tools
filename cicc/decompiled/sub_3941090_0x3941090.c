// Function: sub_3941090
// Address: 0x3941090
//
__int64 __fastcall sub_3941090(_QWORD *a1)
{
  unsigned __int64 *v2; // rdi
  __m128i *v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  const __m128i *v8; // r15
  unsigned __int64 v9; // r13
  __int64 result; // rax
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdx
  unsigned int v13; // edi
  unsigned __int64 v14; // rcx
  unsigned int v15; // eax
  unsigned __int64 v16; // rsi
  _BYTE *v17; // rdi
  unsigned __int32 v18; // esi
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  char v21; // r9
  __int64 v22; // rcx
  unsigned __int64 v23; // r13
  __m128i *v24; // rbx
  __int64 v25; // rax
  const __m128i *v26; // r15
  __m128i *v27; // r14
  const __m128i *v28; // rax
  __m128i **v29; // [rsp+0h] [rbp-B0h]
  _QWORD *v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v32; // [rsp+18h] [rbp-98h]
  unsigned __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  char v34; // [rsp+30h] [rbp-80h]
  unsigned __int64 v35; // [rsp+40h] [rbp-70h] BYREF
  char v36; // [rsp+50h] [rbp-60h]
  __m128i v37; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v38[8]; // [rsp+70h] [rbp-40h] BYREF

  v2 = &v33;
  v3 = (__m128i *)a1;
  sub_393FF90((__int64)&v33, a1);
  if ( (v34 & 1) == 0 || (result = (unsigned int)v33, !(_DWORD)v33) )
  {
    v7 = v33;
    v29 = (__m128i **)(a1 + 11);
    if ( v33 > 0x3FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::reserve");
    v8 = (const __m128i *)a1[11];
    if ( v33 > (__int64)(a1[13] - (_QWORD)v8) >> 5 )
    {
      v23 = a1[12];
      v24 = 0;
      v32 = v23 - (_QWORD)v8;
      v31 = 2 * v33;
      if ( v33 )
      {
        v2 = (unsigned __int64 *)(32 * v33);
        v25 = sub_22077B0(32 * v33);
        v23 = a1[12];
        v8 = (const __m128i *)a1[11];
        v24 = (__m128i *)v25;
      }
      if ( v8 != (const __m128i *)v23 )
      {
        v26 = v8 + 1;
        v27 = v24;
        while ( 1 )
        {
          if ( v27 )
          {
            v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
            v28 = (const __m128i *)v26[-1].m128i_i64[0];
            if ( v28 == v26 )
            {
              v27[1] = _mm_loadu_si128(v26);
            }
            else
            {
              v27->m128i_i64[0] = (__int64)v28;
              v27[1].m128i_i64[0] = v26->m128i_i64[0];
            }
            v27->m128i_i64[1] = v26[-1].m128i_i64[1];
            v26[-1].m128i_i64[0] = (__int64)v26;
          }
          else
          {
            v2 = (unsigned __int64 *)v26[-1].m128i_i64[0];
            if ( v2 != (unsigned __int64 *)v26 )
            {
              v3 = (__m128i *)(v26->m128i_i64[0] + 1);
              j_j___libc_free_0((unsigned __int64)v2);
            }
          }
          v27 += 2;
          if ( (const __m128i *)v23 == &v26[1] )
            break;
          v26 += 2;
        }
        v23 = a1[11];
      }
      if ( v23 )
      {
        v2 = (unsigned __int64 *)v23;
        v3 = (__m128i *)(a1[13] - v23);
        j_j___libc_free_0(v23);
      }
      a1[11] = v24;
      v7 = v33;
      a1[12] = (char *)v24 + v32;
      a1[13] = &v24[v31];
    }
    LODWORD(v9) = 0;
    if ( !v7 )
    {
LABEL_26:
      sub_393D180((__int64)v2, (__int64)v3, v7, v4, v5, v6);
      return 0;
    }
    v30 = a1;
    while ( 1 )
    {
      sub_393FF90((__int64)&v35, v30);
      if ( (v36 & 1) != 0 )
      {
        result = (unsigned int)v35;
        if ( (_DWORD)v35 )
          return result;
      }
      v11 = v35;
      if ( v35 <= 9 )
      {
        v37.m128i_i64[0] = (__int64)v38;
        sub_2240A50(v37.m128i_i64, 1u, 0);
        v17 = (_BYTE *)v37.m128i_i64[0];
      }
      else
      {
        if ( v35 <= 0x63 )
        {
          v37.m128i_i64[0] = (__int64)v38;
          sub_2240A50(v37.m128i_i64, 2u, 0);
          v17 = (_BYTE *)v37.m128i_i64[0];
LABEL_30:
          v17[1] = a00010203040506_0[2 * v11 + 1];
          *v17 = a00010203040506_0[2 * v11];
          goto LABEL_23;
        }
        if ( v35 <= 0x3E7 )
        {
          v16 = 3;
        }
        else if ( v35 <= 0x270F )
        {
          v16 = 4;
        }
        else
        {
          v12 = v35;
          v13 = 1;
          do
          {
            v14 = v12;
            v15 = v13;
            v13 += 4;
            v12 /= 0x2710u;
            if ( v14 <= 0x1869F )
            {
              v16 = v13;
              goto LABEL_18;
            }
            if ( v14 <= 0xF423F )
            {
              v37.m128i_i64[0] = (__int64)v38;
              v16 = v15 + 5;
              goto LABEL_19;
            }
            if ( v14 <= (unsigned __int64)&loc_98967F )
            {
              v16 = v15 + 6;
              goto LABEL_18;
            }
          }
          while ( v14 > 0x5F5E0FF );
          v16 = v15 + 7;
        }
LABEL_18:
        v37.m128i_i64[0] = (__int64)v38;
LABEL_19:
        sub_2240A50(v37.m128i_i64, v16, 0);
        v17 = (_BYTE *)v37.m128i_i64[0];
        v18 = v37.m128i_i32[2] - 1;
        do
        {
          v19 = v11
              - 20
              * (v11 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v11 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
          v20 = v11;
          v11 /= 0x64u;
          v21 = a00010203040506_0[2 * v19 + 1];
          LOBYTE(v19) = a00010203040506_0[2 * v19];
          v17[v18] = v21;
          v22 = v18 - 1;
          v18 -= 2;
          v17[v22] = v19;
        }
        while ( v20 > 0x270F );
        if ( v20 > 0x3E7 )
          goto LABEL_30;
      }
      *v17 = v11 + 48;
LABEL_23:
      v3 = &v37;
      sub_8F9C20(v29, &v37);
      v2 = (unsigned __int64 *)v37.m128i_i64[0];
      if ( (_QWORD *)v37.m128i_i64[0] != v38 )
      {
        v3 = (__m128i *)(v38[0] + 1LL);
        j_j___libc_free_0(v37.m128i_u64[0]);
      }
      v9 = (unsigned int)(v9 + 1);
      if ( v9 >= v33 )
        goto LABEL_26;
    }
  }
  return result;
}
