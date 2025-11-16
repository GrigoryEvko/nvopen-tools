// Function: sub_2253D60
// Address: 0x2253d60
//
__int64 __fastcall sub_2253D60(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v7; // r11d
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  char v11; // bp
  char *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __m128i v15; // xmm0
  int v16; // eax
  const char *v18; // rdi
  const char *v19; // rsi
  int v20; // eax
  __int64 v21; // [rsp+0h] [rbp-78h]
  unsigned __int8 v22; // [rsp+0h] [rbp-78h]
  __int32 v23; // [rsp+Ch] [rbp-6Ch]
  __m128i v25; // [rsp+20h] [rbp-58h] BYREF
  __int64 v26; // [rsp+30h] [rbp-48h]

  v7 = sub_22522E0(a1, a2, (__int64)a3, a4);
  if ( !(_BYTE)v7 )
  {
    v23 = *(_DWORD *)(a4 + 12);
    if ( (v23 & 0x10) != 0 )
      v23 = *(_DWORD *)(a1 + 16);
    v8 = *(unsigned int *)(a1 + 20);
    if ( *(_DWORD *)(a1 + 20) )
    {
      v9 = a1 + 16 * v8;
      do
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)(v9 + 16);
          v25.m128i_i64[0] = 0;
          v25.m128i_i32[2] = 0;
          v25.m128i_i32[3] = v23;
          v26 = 0;
          v11 = v10 & 1;
          v21 = v10 & 2;
          if ( (v10 & 2) == 0 && (v23 & 1) == 0 )
            goto LABEL_9;
          v12 = 0;
          if ( a3 )
          {
            v13 = v10 >> 8;
            if ( v11 )
              v13 = *(_QWORD *)(*a3 + v13);
            v12 = (char *)a3 + v13;
          }
          v7 = (*(__int64 (__fastcall **)(_QWORD, __int64, char *, __m128i *))(**(_QWORD **)(v9 + 8) + 48LL))(
                 *(_QWORD *)(v9 + 8),
                 a2,
                 v12,
                 &v25);
          if ( !(_BYTE)v7 )
            goto LABEL_9;
          if ( v26 == 16 && v11 )
            v26 = *(_QWORD *)(v9 + 8);
          if ( !v21 && v25.m128i_i32[2] > 3 )
            v25.m128i_i32[2] &= ~2u;
          v14 = *(_QWORD *)(a4 + 16);
          if ( !v14 )
            break;
          if ( *(_QWORD *)a4 != v25.m128i_i64[0] )
          {
            *(_QWORD *)a4 = 0;
            *(_DWORD *)(a4 + 8) = 2;
            return v7;
          }
          if ( !*(_QWORD *)a4 )
          {
            if ( v14 == 16
              || v26 == 16
              || (v18 = *(const char **)(v26 + 8), v19 = *(const char **)(v14 + 8), v18 != v19)
              && (*v18 == 42 || (v22 = v7, v20 = strcmp(v18, v19), v7 = v22, v20)) )
            {
              *(_DWORD *)(a4 + 8) = 2;
              return v7;
            }
          }
          *(_DWORD *)(a4 + 8) |= v25.m128i_u32[2];
LABEL_9:
          v9 -= 16;
          if ( !--v8 )
            goto LABEL_37;
        }
        v15 = _mm_loadu_si128(&v25);
        *(_QWORD *)(a4 + 16) = v26;
        *(__m128i *)a4 = v15;
        v16 = *(_DWORD *)(a4 + 8);
        if ( v16 <= 3 )
          return v7;
        if ( (v16 & 2) != 0 )
        {
          if ( (*(_BYTE *)(a1 + 16) & 1) == 0 )
            return v7;
          goto LABEL_9;
        }
        if ( (v16 & 1) == 0 || (*(_BYTE *)(a1 + 16) & 2) == 0 )
          return v7;
        v9 -= 16;
        --v8;
      }
      while ( v8 );
    }
LABEL_37:
    LOBYTE(v7) = *(_DWORD *)(a4 + 8) != 0;
  }
  return v7;
}
