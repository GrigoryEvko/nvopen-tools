// Function: sub_31C9280
// Address: 0x31c9280
//
unsigned __int64 __fastcall sub_31C9280(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rbp
  __int64 v8; // r12
  unsigned __int64 result; // rax
  __int64 v12; // r12
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r12
  __m128i *v16; // rsi
  __m128i *v17; // rsi
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // r14
  __int64 v21; // rax
  _DWORD *v22; // rdi
  __int64 v23; // rcx
  __m128i *v24; // rsi
  __m128i v25[3]; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v26; // [rsp-78h] [rbp-78h] BYREF
  __m128i v27; // [rsp-68h] [rbp-68h] BYREF
  __m128i v28; // [rsp-58h] [rbp-58h] BYREF
  __m128i v29; // [rsp-48h] [rbp-48h] BYREF
  unsigned __int64 v30; // [rsp-38h] [rbp-38h]
  char v31; // [rsp-30h] [rbp-30h]
  __int64 v32; // [rsp-28h] [rbp-28h]
  __int64 v33; // [rsp-20h] [rbp-20h]
  __int64 v34; // [rsp-8h] [rbp-8h]

  result = (unsigned int)*a2 - 29;
  v34 = v7;
  v33 = v8;
  v32 = v6;
  switch ( (int)result )
  {
    case 0:
      BUG();
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
    case 35:
    case 36:
    case 37:
    case 38:
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
    case 44:
    case 45:
    case 46:
    case 47:
    case 48:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
      goto LABEL_2;
    case 32:
      result = (unsigned __int64)sub_31C8800(&v26, (__int64 *)a1, (__int64)a2);
      if ( v31 )
      {
        v17 = *(__m128i **)(a1 + 144);
        if ( v17 == *(__m128i **)(a1 + 152) )
        {
          return sub_31C9090((unsigned __int64 *)(a1 + 136), v17, &v26);
        }
        else
        {
          if ( v17 )
          {
            *v17 = _mm_loadu_si128(&v26);
            v17[1] = _mm_loadu_si128(&v27);
            v17[2] = _mm_loadu_si128(&v28);
            v17[3] = _mm_loadu_si128(&v29);
            result = v30;
            v17[4].m128i_i64[0] = v30;
            v17 = *(__m128i **)(a1 + 144);
          }
          *(_QWORD *)(a1 + 144) = (char *)v17 + 72;
        }
      }
      return result;
    case 33:
      if ( !(unsigned int)sub_BD2910(*(_QWORD *)(a1 + 344)) )
        goto LABEL_2;
      if ( *(_QWORD *)(a1 + 48) )
        goto LABEL_21;
      v19 = sub_31C85D0(a1, (__int64)a2);
      if ( !v19 )
        goto LABEL_21;
      v20 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL);
      v21 = sub_B2BD20(v19);
      if ( *(_BYTE *)(v21 + 8) != 15 || *(_BYTE *)(v20 + 8) != 15 )
      {
        if ( v20 != v21 )
          goto LABEL_21;
        goto LABEL_47;
      }
      if ( *(_DWORD *)(v21 + 12) && *(_DWORD *)(v20 + 12) && **(_QWORD **)(v21 + 16) == **(_QWORD **)(v20 + 16) )
      {
LABEL_47:
        *(_QWORD *)(a1 + 48) = a2;
        sub_D66630(v25, (__int64)a2);
        v22 = (_DWORD *)(a1 + 56);
        v23 = 12;
        v24 = v25;
        while ( v23 )
        {
          *v22 = v24->m128i_i32[0];
          v24 = (__m128i *)((char *)v24 + 4);
          ++v22;
          --v23;
        }
        *(_QWORD *)(a1 + 104) = v19;
      }
LABEL_21:
      result = (unsigned __int64)sub_31C8800(&v26, (__int64 *)a1, (__int64)a2);
      if ( v31 )
      {
        v16 = *(__m128i **)(a1 + 120);
        if ( v16 == *(__m128i **)(a1 + 128) )
        {
          return sub_31C9090((unsigned __int64 *)(a1 + 112), v16, &v26);
        }
        else
        {
          if ( v16 )
          {
            *v16 = _mm_loadu_si128(&v26);
            v16[1] = _mm_loadu_si128(&v27);
            v16[2] = _mm_loadu_si128(&v28);
            v16[3] = _mm_loadu_si128(&v29);
            result = v30;
            v16[4].m128i_i64[0] = v30;
          }
          *(_QWORD *)(a1 + 120) += 72LL;
        }
      }
      return result;
    case 34:
      result = sub_BD2910(*(_QWORD *)(a1 + 344));
      if ( (_DWORD)result )
      {
LABEL_2:
        result = *(_QWORD *)(a1 + 344);
        *(_QWORD *)(a1 + 336) = result;
      }
      else
      {
        v15 = *((_QWORD *)a2 + 2);
        if ( v15 )
        {
          result = *(unsigned int *)(a1 + 168);
          do
          {
            if ( result + 1 > *(unsigned int *)(a1 + 172) )
            {
              sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), result + 1, 8u, v13, v14);
              result = *(unsigned int *)(a1 + 168);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = v15;
            result = (unsigned int)(*(_DWORD *)(a1 + 168) + 1);
            *(_DWORD *)(a1 + 168) = result;
            v15 = *(_QWORD *)(v15 + 8);
          }
          while ( v15 );
        }
      }
      return result;
    case 49:
      v12 = *((_QWORD *)a2 + 2);
      if ( v12 )
      {
        result = *(unsigned int *)(a1 + 168);
        do
        {
          if ( result + 1 > *(unsigned int *)(a1 + 172) )
          {
            sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), result + 1, 8u, a5, a6);
            result = *(unsigned int *)(a1 + 168);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = v12;
          result = (unsigned int)(*(_DWORD *)(a1 + 168) + 1);
          *(_DWORD *)(a1 + 168) = result;
          v12 = *(_QWORD *)(v12 + 8);
        }
        while ( v12 );
      }
      return result;
    case 55:
      v18 = *((_QWORD *)a2 + 2);
      if ( v18 )
      {
        result = *(unsigned int *)(a1 + 168);
        do
        {
          if ( result + 1 > *(unsigned int *)(a1 + 172) )
          {
            sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), result + 1, 8u, a5, a6);
            result = *(unsigned int *)(a1 + 168);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = v18;
          result = (unsigned int)(*(_DWORD *)(a1 + 168) + 1);
          *(_DWORD *)(a1 + 168) = result;
          v18 = *(_QWORD *)(v18 + 8);
        }
        while ( v18 );
      }
      return result;
    case 56:
      result = *((_QWORD *)a2 - 4);
      if ( result )
      {
        if ( !*(_BYTE *)result && *(_QWORD *)(result + 24) == *((_QWORD *)a2 + 10) )
        {
          result = *(unsigned int *)(result + 36);
          if ( (unsigned int)result <= 0xF5 && (unsigned int)result > 0xED )
            result = (unsigned int)(result - 238);
        }
      }
      *(_QWORD *)(a1 + 336) = *(_QWORD *)(a1 + 344);
      return result;
    default:
      BUG();
  }
}
