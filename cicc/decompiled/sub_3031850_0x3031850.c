// Function: sub_3031850
// Address: 0x3031850
//
_QWORD *__fastcall sub_3031850(_QWORD *a1, __int64 a2, _QWORD *a3, char a4, __int64 a5)
{
  _DWORD *v6; // rsi
  _QWORD *v7; // r11
  _QWORD *v8; // r10
  char v10; // bl
  unsigned __int64 v11; // r13
  _DWORD *v12; // rax
  char v13; // cl
  _QWORD *v14; // r12
  unsigned int v15; // r14d
  _QWORD *v16; // r11
  unsigned int *v17; // r13
  unsigned int v18; // r10d
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // ecx
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned int v27; // r10d
  __int64 v28; // rdx
  unsigned int v29; // ebx
  __int64 v31; // rax
  _DWORD *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // [rsp+20h] [rbp-A0h]
  unsigned int v35; // [rsp+28h] [rbp-98h]
  unsigned int v36; // [rsp+28h] [rbp-98h]
  __int64 v37; // [rsp+28h] [rbp-98h]
  __int64 v38; // [rsp+30h] [rbp-90h]
  int v39; // [rsp+44h] [rbp-7Ch]
  unsigned __int64 v40; // [rsp+48h] [rbp-78h]
  __int64 v42; // [rsp+50h] [rbp-70h]
  __m128i v43; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v44; // [rsp+70h] [rbp-50h] BYREF
  char v45; // [rsp+78h] [rbp-48h]
  __int64 v46; // [rsp+80h] [rbp-40h]
  __int64 v47; // [rsp+88h] [rbp-38h]

  v6 = a1 + 2;
  v7 = a1;
  v8 = a3;
  v10 = a5;
  *a1 = a1 + 2;
  a1[1] = 0x1000000000LL;
  v11 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v11 > 0x10 )
  {
    v42 = a2;
    sub_C8D5F0((__int64)a1, v6, v11, 4u, a5, a2);
    v7 = a1;
    a2 = v42;
    v8 = a3;
    v32 = (_DWORD *)*a1;
    v33 = *a1 + 4 * v11;
    do
      *v32++ = 3;
    while ( (_DWORD *)v33 != v32 );
  }
  else if ( *(_DWORD *)(a2 + 8) )
  {
    v12 = &v6[v11];
    do
      *v6++ = 3;
    while ( v12 != v6 );
  }
  *((_DWORD *)v7 + 2) = v11;
  if ( !v10 )
  {
    v39 = *(_DWORD *)(a2 + 8);
    if ( v39 )
    {
      v13 = a4;
      v14 = v7;
      v15 = 0;
      v34 = a2;
      v16 = v8;
      v40 = 1LL << v13;
      do
      {
        v38 = v15 + 1;
        v17 = (unsigned int *)&unk_44C7AD0;
        v18 = 16;
        v19 = v16;
        while ( 1 )
        {
          if ( v18 <= v40 && (*(_QWORD *)(*v19 + 8LL * v15) & (v18 - 1)) == 0 )
          {
            v43 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)v34 + 16LL * v15));
            if ( v43.m128i_i16[0] )
            {
              if ( v43.m128i_i16[0] == 1 || (unsigned __int16)(v43.m128i_i16[0] - 504) <= 7u )
LABEL_42:
                BUG();
              v20 = *(_QWORD *)&byte_444C4A0[16 * v43.m128i_u16[0] - 16];
              LOBYTE(v21) = byte_444C4A0[16 * v43.m128i_u16[0] - 8];
            }
            else
            {
              v35 = v18;
              v20 = sub_3007260((__int64)&v43);
              v18 = v35;
              v46 = v20;
              v47 = v21;
            }
            v36 = v18;
            v45 = v21;
            v44 = (unsigned __int64)(v20 + 7) >> 3;
            v22 = sub_CA1930(&v44);
            v23 = v22;
            if ( v36 > v22 )
            {
              v24 = v36 / v22;
              if ( v36 == v23 * (v36 / v23) && v15 + v24 <= *(_DWORD *)(v34 + 8) )
              {
                v25 = (unsigned int)(v24 - 2);
                if ( (v25 & 0xFFFFFFFD) == 0 )
                {
                  if ( v15 + v24 <= v15 + 1 )
                  {
LABEL_23:
                    if ( v24 == 2 )
                    {
                      v16 = v19;
                      v29 = v15 + 2;
                      *(_DWORD *)(*v14 + 4LL * (int)v15) = 1;
                      *(_DWORD *)(*v14 + 4LL * (int)v15 + 4) = 2;
                      goto LABEL_25;
                    }
                    if ( v24 == 4 )
                    {
                      v16 = v19;
                      v29 = v15 + 4;
                      v31 = 4LL * (int)v15;
                      *(_DWORD *)(*v14 + v31) = 1;
                      *(_DWORD *)(*v14 + v31 + 4) = 0;
                      *(_DWORD *)(*v14 + v31 + 8) = 0;
                      *(_DWORD *)(*v14 + v31 + 12) = 2;
                      goto LABEL_25;
                    }
                    if ( v24 != 1 )
                      goto LABEL_42;
                  }
                  else
                  {
                    v26 = *(_QWORD *)v34;
                    v27 = v15;
                    v37 = 8 * (v38 + v25 + 1);
                    v28 = 8 * v38;
                    while ( v43.m128i_i16[0] == *(_WORD *)(v26 + 2 * v28)
                         && (v43.m128i_i16[0] || v43.m128i_i64[1] == *(_QWORD *)(v26 + 2 * v28 + 8))
                         && *(_QWORD *)(*v19 + v28) - *(_QWORD *)(*v19 + 8LL * v27) == v23 )
                    {
                      ++v27;
                      v28 += 8;
                      if ( v37 == v28 )
                        goto LABEL_23;
                    }
                  }
                }
              }
            }
          }
          if ( &unk_44C7AE0 == (_UNKNOWN *)++v17 )
            break;
          v18 = *v17;
        }
        v29 = v15 + 1;
        v16 = v19;
LABEL_25:
        v15 = v29;
      }
      while ( v39 != v29 );
      return v14;
    }
  }
  return v7;
}
