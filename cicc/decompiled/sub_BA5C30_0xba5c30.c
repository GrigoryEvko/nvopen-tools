// Function: sub_BA5C30
// Address: 0xba5c30
//
void __fastcall sub_BA5C30(const __m128i *a1, __int64 *m128i_i64)
{
  __int64 *v3; // rbx
  unsigned __int32 v4; // eax
  const __m128i *v5; // r13
  const __m128i *v6; // r15
  __m128i *v7; // r15
  const __m128i *v8; // rcx
  __int64 v9; // r8
  const __m128i *v10; // rdx
  __m128i *v11; // rcx
  const __m128i *v12; // rax
  __int64 v13; // r13
  __m128i *v14; // r12
  unsigned __int64 v15; // rax
  char *v16; // r13
  const __m128i *v17; // rdi
  int v18; // edx
  unsigned int v19; // eax
  __int64 v20; // rax
  __m128i *v21; // rdi
  int v22; // eax
  __int64 *v23; // r8
  __int32 v24; // edx
  __m128i *v25; // r13
  __m128i *v26; // rdi
  const __m128i *v27; // r8
  int v28; // edx
  unsigned int v29; // eax
  __int64 **v30; // rdi
  __int64 *v31; // r9
  unsigned __int32 v32; // eax
  __int32 v33; // edx
  int v34; // r9d
  int v35; // edi
  int v36; // r10d
  int v37; // [rsp+0h] [rbp-110h]
  __m128i *v38; // [rsp+10h] [rbp-100h] BYREF
  __int64 v39; // [rsp+18h] [rbp-F8h]
  _BYTE v40[240]; // [rsp+20h] [rbp-F0h] BYREF

  v3 = m128i_i64;
  v4 = (unsigned __int32)a1[1].m128i_i32[2] >> 1;
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
    v5 = a1 + 8;
    v6 = a1 + 2;
    if ( v4 )
      goto LABEL_10;
    goto LABEL_3;
  }
  v6 = (const __m128i *)a1[2].m128i_i64[0];
  v5 = (const __m128i *)((char *)v6 + 24 * a1[2].m128i_u32[2]);
  if ( !v4 )
    goto LABEL_3;
  while ( 1 )
  {
LABEL_10:
    if ( v6 == v5 )
      goto LABEL_3;
    if ( v6->m128i_i64[0] != -4096 && v6->m128i_i64[0] != -8192 )
      break;
    v6 = (const __m128i *)((char *)v6 + 24);
  }
  v38 = (__m128i *)v40;
  v39 = 0x800000000LL;
  if ( v6 == v5 )
  {
LABEL_3:
    v7 = (__m128i *)v40;
    goto LABEL_4;
  }
  v8 = v6;
  v9 = 0;
  while ( 1 )
  {
    v10 = (const __m128i *)((char *)v8 + 24);
    if ( &v8[1].m128i_u64[1] == (unsigned __int64 *)v5 )
      break;
    while ( 1 )
    {
      v8 = v10;
      if ( v10->m128i_i64[0] != -8192 && v10->m128i_i64[0] != -4096 )
        break;
      v10 = (const __m128i *)((char *)v10 + 24);
      if ( v5 == v10 )
        goto LABEL_18;
    }
    ++v9;
    if ( v10 == v5 )
      goto LABEL_19;
  }
LABEL_18:
  ++v9;
LABEL_19:
  v11 = (__m128i *)v40;
  if ( v9 > 8 )
  {
    m128i_i64 = (__int64 *)v40;
    v37 = v9;
    sub_C8D5F0(&v38, v40, v9, 24);
    LODWORD(v9) = v37;
    v11 = (__m128i *)((char *)v38 + 24 * (unsigned int)v39);
  }
  do
  {
    if ( v11 )
    {
      *v11 = _mm_loadu_si128(v6);
      v11[1].m128i_i64[0] = v6[1].m128i_i64[0];
    }
    v12 = (const __m128i *)((char *)v6 + 24);
    if ( &v6[1].m128i_u64[1] == (unsigned __int64 *)v5 )
      break;
    while ( 1 )
    {
      v6 = v12;
      if ( v12->m128i_i64[0] != -8192 && v12->m128i_i64[0] != -4096 )
        break;
      v12 = (const __m128i *)((char *)v12 + 24);
      if ( v5 == v12 )
        goto LABEL_27;
    }
    v11 = (__m128i *)((char *)v11 + 24);
  }
  while ( v12 != v5 );
LABEL_27:
  v7 = v38;
  LODWORD(v39) = v39 + v9;
  v13 = 24LL * (unsigned int)v39;
  v14 = (__m128i *)((char *)v38 + v13);
  if ( &v38->m128i_i8[v13] != (__int8 *)v38 )
  {
    _BitScanReverse64(&v15, 0xAAAAAAAAAAAAAAABLL * (v13 >> 3));
    sub_B8F080(
      v38->m128i_i8,
      (__m128i *)((char *)v38 + v13),
      2LL * (int)(63 - (v15 ^ 0x3F)),
      (__int64)v11,
      (unsigned int)v39);
    if ( (unsigned __int64)v13 > 0x180 )
    {
      v25 = v7 + 24;
      m128i_i64 = v7[24].m128i_i64;
      sub_B8E6F0(v7, v7 + 24);
      if ( v14 != &v7[24] )
      {
        do
        {
          v26 = v25;
          v25 = (__m128i *)((char *)v25 + 24);
          sub_B8E410(v26);
        }
        while ( v14 != v25 );
      }
    }
    else
    {
      m128i_i64 = (__int64 *)v14;
      sub_B8E6F0(v7, v14);
    }
    v7 = v38;
    v16 = &v38->m128i_i8[24 * (unsigned int)v39];
    if ( v16 != (char *)v38 )
    {
      while ( 1 )
      {
        v23 = (__int64 *)v7->m128i_i64[0];
        if ( (a1[1].m128i_i8[8] & 1) != 0 )
        {
          v17 = a1 + 2;
          v18 = 3;
        }
        else
        {
          v24 = a1[2].m128i_i32[2];
          v17 = (const __m128i *)a1[2].m128i_i64[0];
          if ( !v24 )
            goto LABEL_38;
          v18 = v24 - 1;
        }
        v19 = v18 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        m128i_i64 = (__int64 *)v17->m128i_i64[3 * v19];
        if ( v23 != m128i_i64 )
        {
          v34 = 1;
          while ( m128i_i64 != (__int64 *)-4096LL )
          {
            v19 = v18 & (v34 + v19);
            m128i_i64 = (__int64 *)v17->m128i_i64[3 * v19];
            if ( v23 == m128i_i64 )
              goto LABEL_34;
            ++v34;
          }
          goto LABEL_38;
        }
LABEL_34:
        v20 = v7->m128i_i64[1];
        v21 = (__m128i *)(v20 & 0xFFFFFFFFFFFFFFFCLL);
        if ( (v20 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
        {
          v22 = v20 & 3;
          if ( v22 )
          {
            if ( v22 == 2 )
            {
              m128i_i64 = (__int64 *)v7->m128i_i64[0];
              sub_B98CB0((__int64)v21, (unsigned __int8 **)v7->m128i_i64[0], v3);
            }
            else
            {
              m128i_i64 = (__int64 *)v7->m128i_i64[0];
              switch ( v21->m128i_i8[0] )
              {
                case 4:
                  sub_B00E90((__int64)v21, m128i_i64, (__int64)v3);
                  break;
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 0xA:
                case 0xB:
                case 0xC:
                case 0xD:
                case 0xE:
                case 0xF:
                case 0x10:
                case 0x11:
                case 0x12:
                case 0x13:
                case 0x14:
                case 0x15:
                case 0x16:
                case 0x17:
                case 0x18:
                case 0x19:
                case 0x1A:
                case 0x1B:
                case 0x1C:
                case 0x1D:
                case 0x1E:
                case 0x1F:
                case 0x20:
                case 0x21:
                case 0x22:
                case 0x23:
                case 0x24:
                  sub_BA56C0(v21, (__int64)m128i_i64, (unsigned __int8 *)v3);
                  break;
                default:
                  BUG();
              }
            }
          }
          else
          {
            m128i_i64 = v3;
            sub_B9F930((__int64)v21, v3);
          }
          goto LABEL_38;
        }
        *v23 = (__int64)v3;
        if ( v3 )
        {
          m128i_i64 = v3;
          sub_B96E90((__int64)v23, (__int64)v3, 1);
        }
        if ( (a1[1].m128i_i8[8] & 1) != 0 )
        {
          v27 = a1 + 2;
          v28 = 3;
          goto LABEL_55;
        }
        v33 = a1[2].m128i_i32[2];
        v27 = (const __m128i *)a1[2].m128i_i64[0];
        if ( v33 )
        {
          v28 = v33 - 1;
LABEL_55:
          m128i_i64 = (__int64 *)v7->m128i_i64[0];
          v29 = v28 & (((unsigned int)v7->m128i_i64[0] >> 9) ^ ((unsigned int)v7->m128i_i64[0] >> 4));
          v30 = (__int64 **)v27 + 3 * v29;
          v31 = *v30;
          if ( (__int64 *)v7->m128i_i64[0] != *v30 )
          {
            v35 = 1;
            while ( v31 != (__int64 *)-4096LL )
            {
              v36 = v35 + 1;
              v29 = v28 & (v35 + v29);
              v30 = (__int64 **)v27 + 3 * v29;
              v31 = *v30;
              if ( m128i_i64 == *v30 )
                goto LABEL_56;
              v35 = v36;
            }
            goto LABEL_38;
          }
LABEL_56:
          *v30 = (__int64 *)-8192LL;
          v32 = a1[1].m128i_u32[2];
          v7 = (__m128i *)((char *)v7 + 24);
          ++a1[1].m128i_i32[3];
          a1[1].m128i_i32[2] = (2 * (v32 >> 1) - 2) | v32 & 1;
          if ( v16 == (char *)v7 )
          {
LABEL_57:
            v7 = v38;
            break;
          }
        }
        else
        {
LABEL_38:
          v7 = (__m128i *)((char *)v7 + 24);
          if ( v16 == (char *)v7 )
            goto LABEL_57;
        }
      }
    }
  }
LABEL_4:
  if ( v7 != (__m128i *)v40 )
    _libc_free(v7, m128i_i64);
}
