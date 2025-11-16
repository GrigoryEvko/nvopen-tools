// Function: sub_F29590
// Address: 0xf29590
//
unsigned __int8 *__fastcall sub_F29590(__m128i *a1, __int64 a2)
{
  __int64 *v2; // r13
  _BYTE *v4; // r15
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __m128i v7; // xmm3
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 *v11; // r9
  __int64 v12; // rdx
  _BYTE *v14; // r14
  _BYTE *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rbx
  _BYTE *v18; // r15
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rbx
  _BYTE *v24; // rdx
  _BYTE *v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // [rsp+0h] [rbp-100h]
  _BYTE *v32; // [rsp+0h] [rbp-100h]
  char v33; // [rsp+8h] [rbp-F8h]
  __int64 v34; // [rsp+8h] [rbp-F8h]
  _QWORD v35[2]; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE *v36; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v37; // [rsp+28h] [rbp-D8h]
  _BYTE v38[64]; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v39; // [rsp+70h] [rbp-90h] BYREF
  __m128i v40; // [rsp+80h] [rbp-80h]
  _QWORD v41[2]; // [rsp+90h] [rbp-70h] BYREF
  __m128i v42; // [rsp+A0h] [rbp-60h]
  __int64 v43; // [rsp+B0h] [rbp-50h]

  v2 = (__int64 *)a2;
  v4 = *(_BYTE **)(a2 - 32);
  v5 = _mm_loadu_si128(a1 + 6);
  v6 = _mm_loadu_si128(a1 + 7);
  v7 = _mm_loadu_si128(a1 + 9);
  v8 = a1[10].m128i_i64[0];
  v41[0] = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v41[1] = a2;
  v43 = v8;
  v39 = v5;
  v40 = v6;
  v42 = v7;
  v9 = sub_1002A90(v4, &v39);
  if ( v9 )
  {
LABEL_2:
    v12 = v9;
    return sub_F162A0((__int64)a1, (__int64)v2, v12);
  }
  if ( *v4 != 84
    || (v14 = sub_F27020((__int64)a1, a2, (__int64)v4, 0, v10, v11)) == 0
    && (v14 = sub_F26680((__int64)a1, a2, (__int64)v4)) == 0 )
  {
    v14 = (_BYTE *)sub_F292F0((__int64)a1, a2);
    if ( v14 )
    {
LABEL_6:
      v12 = (__int64)v14;
      return sub_F162A0((__int64)a1, (__int64)v2, v12);
    }
    if ( (unsigned __int8)(*v4 - 12) <= 1u )
      goto LABEL_67;
    if ( (unsigned __int8)(*v4 - 9) > 2u )
      goto LABEL_68;
    a2 = (__int64)v4;
    v40.m128i_i8[12] = 1;
    v35[0] = &v39;
    v39.m128i_i64[1] = (__int64)v41;
    v36 = v38;
    v39.m128i_i64[0] = 0;
    v40.m128i_i64[0] = 8;
    v40.m128i_i32[2] = 0;
    v37 = 0x800000000LL;
    v35[1] = &v36;
    v33 = sub_AA8FD0(v35, (__int64)v4);
    if ( v33 )
    {
      while ( 1 )
      {
        v15 = v36;
        if ( !(_DWORD)v37 )
          break;
        a2 = *(_QWORD *)&v36[8 * (unsigned int)v37 - 8];
        LODWORD(v37) = v37 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(v35, a2) )
          goto LABEL_64;
      }
    }
    else
    {
LABEL_64:
      v33 = 0;
      v15 = v36;
    }
    if ( v15 != v38 )
      _libc_free(v15, a2);
    if ( !v40.m128i_i8[12] )
      _libc_free(v39.m128i_i64[1], a2);
    if ( v33 )
    {
LABEL_67:
      if ( !(unsigned __int8)sub_F08290((__int64)v2) )
      {
        v34 = v2[1];
        v16 = sub_AD6530(v34, a2);
        v17 = v2[2];
        v18 = (_BYTE *)v16;
        if ( v17 )
        {
          do
          {
            v20 = *(_QWORD *)(v17 + 24);
            if ( *(_BYTE *)v20 == 58 )
            {
              v19 = sub_AD62B0(v34);
            }
            else if ( *(_BYTE *)v20 == 86 )
            {
              if ( (*(_BYTE *)(v20 + 7) & 0x40) != 0 )
                v30 = *(_QWORD *)(v20 - 8);
              else
                v30 = v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF);
              v19 = (__int64)v18;
              if ( v2 == *(__int64 **)v30 && **(_BYTE **)(v30 + 32) <= 0x15u )
                v19 = sub_AD6400(v34);
            }
            else
            {
              v19 = (__int64)v18;
            }
            if ( v14 )
            {
              if ( (_BYTE *)v19 != v14 )
                v14 = v18;
            }
            else
            {
              v14 = (_BYTE *)v19;
            }
            v17 = *(_QWORD *)(v17 + 8);
          }
          while ( v17 );
        }
        else
        {
          v14 = 0;
        }
        goto LABEL_6;
      }
    }
    else
    {
LABEL_68:
      if ( *v4 <= 0x15u && (unsigned __int8)sub_AD6C40((__int64)v4) )
      {
        v21 = v2[1];
        if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
          v21 = **(_QWORD **)(v21 + 16);
        v22 = sub_AD6530(v21, v21);
        v23 = v2[2];
        v24 = (_BYTE *)v22;
        if ( v23 )
        {
          do
          {
            v26 = *(_QWORD *)(v23 + 24);
            if ( *(_BYTE *)v26 == 58 )
            {
              v31 = v24;
              v27 = sub_AD62B0(v21);
              v24 = v31;
              v25 = (_BYTE *)v27;
            }
            else if ( *(_BYTE *)v26 == 86 )
            {
              if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
                v28 = *(_QWORD *)(v26 - 8);
              else
                v28 = v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
              v25 = v24;
              if ( v2 == *(__int64 **)v28 && **(_BYTE **)(v28 + 32) <= 0x15u )
              {
                v32 = v24;
                v29 = sub_AD6400(v21);
                v24 = v32;
                v25 = (_BYTE *)v29;
              }
            }
            else
            {
              v25 = v24;
            }
            if ( v14 )
            {
              if ( v25 != v14 )
                v14 = v24;
            }
            else
            {
              v14 = v25;
            }
            v23 = *(_QWORD *)(v23 + 8);
          }
          while ( v23 );
        }
        else
        {
          v14 = 0;
        }
        v9 = sub_AD6D90((__int64)v4, v14);
        goto LABEL_2;
      }
      if ( (unsigned __int8)sub_F10A80((unsigned __int64 *)a1, v2) )
        return (unsigned __int8 *)v2;
    }
  }
  return v14;
}
