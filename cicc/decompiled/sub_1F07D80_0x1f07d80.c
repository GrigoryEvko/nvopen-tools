// Function: sub_1F07D80
// Address: 0x1f07d80
//
void __fastcall sub_1F07D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // r12
  void *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rax
  __int16 v17; // ax
  int v18; // edx
  __int64 v19; // rax
  const __m128i *v20; // rsi
  __m128i *v21; // rdi
  __int64 v22; // rsi
  __m128i v23; // rax
  const __m128i *v24; // rsi
  const __m128i *v25; // rdi
  __int64 v26; // r14
  unsigned __int64 v27; // rdx
  _DWORD *v28; // rsi
  _BYTE *v29; // rax
  int v30; // edx
  unsigned int v31; // r9d
  __int16 v32; // ax
  const __m128i *v33; // rsi
  __m128i v34; // rax
  __m128i v35; // [rsp+30h] [rbp-140h] BYREF
  const __m128i *v36; // [rsp+40h] [rbp-130h] BYREF
  __m128i *v37; // [rsp+48h] [rbp-128h]
  const __m128i *v38; // [rsp+50h] [rbp-120h]
  __int64 v39; // [rsp+60h] [rbp-110h] BYREF
  unsigned __int64 v40[2]; // [rsp+68h] [rbp-108h] BYREF
  _BYTE v41[32]; // [rsp+78h] [rbp-F8h] BYREF
  int v42; // [rsp+98h] [rbp-D8h]
  const __m128i *v43; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i *v44; // [rsp+A8h] [rbp-C8h]
  const __m128i *v45; // [rsp+B0h] [rbp-C0h]
  _BYTE *v46; // [rsp+B8h] [rbp-B8h]
  __int64 v47; // [rsp+C0h] [rbp-B0h]
  _BYTE v48[96]; // [rsp+C8h] [rbp-A8h] BYREF
  unsigned __int64 v49; // [rsp+128h] [rbp-48h]
  int v50; // [rsp+130h] [rbp-40h]

  v3 = a2;
  v6 = *(_QWORD *)(a1 + 16);
  v40[0] = (unsigned __int64)v41;
  v7 = v6 - *(_QWORD *)(a1 + 8);
  v39 = a1;
  v40[1] = 0x800000000LL;
  v42 = 0;
  sub_3945AE0(v40, v7 >> 3);
  v43 = 0;
  v46 = v48;
  v44 = 0;
  v45 = 0;
  v47 = 0x800000000LL;
  v49 = 0;
  v50 = 0;
  v10 = (__int64)(*(_QWORD *)(v39 + 16) - *(_QWORD *)(v39 + 8)) >> 3;
  if ( (_DWORD)v10 )
  {
    v11 = _libc_calloc((unsigned int)v10, 1u);
    if ( !v11 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v11 = 0;
    }
    v49 = (unsigned __int64)v11;
    v50 = v10;
  }
  v12 = a2 + 272 * a3;
  if ( v12 != a2 )
  {
    do
    {
      v13 = *(unsigned int *)(v3 + 192);
      if ( *(_DWORD *)(*(_QWORD *)(v39 + 8) + 8 * v13 + 4) == -1 )
      {
        v14 = *(_QWORD **)(v3 + 112);
        v15 = &v14[2 * *(unsigned int *)(v3 + 120)];
        if ( v14 == v15 )
        {
LABEL_11:
          v38 = 0;
          v16 = *(_QWORD *)(v3 + 8);
          v36 = 0;
          v37 = 0;
          v17 = **(_WORD **)(v16 + 16);
          switch ( v17 )
          {
            case 0:
            case 8:
            case 10:
            case 14:
            case 15:
            case 45:
LABEL_13:
              v18 = 0;
              break;
            default:
              switch ( v17 )
              {
                case 2:
                case 3:
                case 4:
                case 6:
                case 9:
                case 12:
                case 13:
                case 17:
                case 18:
                  goto LABEL_13;
                default:
                  v18 = 1;
                  break;
              }
              break;
          }
          *(_DWORD *)(*(_QWORD *)(v39 + 8) + 8 * v13) = v18;
          v19 = *(_QWORD *)(v3 + 32);
          v35.m128i_i64[0] = v3;
          v20 = v37;
          v35.m128i_i64[1] = v19;
          if ( v37 != v38 )
          {
            if ( v37 )
            {
              *v37 = _mm_loadu_si128(&v35);
              v20 = v37;
            }
            v21 = (__m128i *)&v20[1];
            v37 = (__m128i *)&v20[1];
            goto LABEL_18;
          }
LABEL_56:
          sub_1F07C00(&v36, v37, &v35);
          v21 = v37;
          do
          {
LABEL_18:
            while ( 1 )
            {
              v22 = v21[-1].m128i_i64[0];
              v23.m128i_i64[1] = v21[-1].m128i_i64[1];
              if ( v23.m128i_i64[1] == *(_QWORD *)(v22 + 32) + 16LL * *(unsigned int *)(v22 + 40) )
                break;
              v21[-1].m128i_i64[1] = v23.m128i_i64[1] + 16;
              if ( (*(_QWORD *)v23.m128i_i64[1] & 6) != 0
                || (v23.m128i_i64[0] = *(_QWORD *)v23.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL,
                    v8 = *(unsigned int *)(v23.m128i_i64[0] + 192),
                    (_DWORD)v8 == -1) )
              {
                v21 = v37;
              }
              else
              {
                v8 = *(_QWORD *)(v39 + 8) + 8 * v8;
                if ( *(_DWORD *)(v8 + 4) == -1 )
                {
                  v32 = **(_WORD **)(*(_QWORD *)(v23.m128i_i64[0] + 8) + 16LL);
                  switch ( v32 )
                  {
                    case 0:
                    case 8:
                    case 10:
                    case 14:
                    case 15:
                    case 45:
LABEL_49:
                      v23.m128i_i32[0] = 0;
                      break;
                    default:
                      switch ( v32 )
                      {
                        case 2:
                        case 3:
                        case 4:
                        case 6:
                        case 9:
                        case 12:
                        case 13:
                        case 17:
                        case 18:
                          goto LABEL_49;
                        default:
                          v23.m128i_i32[0] = 1;
                          break;
                      }
                      break;
                  }
                  *(_DWORD *)v8 = v23.m128i_i32[0];
                  v33 = v37;
                  v34.m128i_i64[0] = *(_QWORD *)v23.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
                  v34.m128i_i64[1] = *(_QWORD *)(v34.m128i_i64[0] + 32);
                  v35 = v34;
                  if ( v37 == v38 )
                    goto LABEL_56;
                  if ( v37 )
                  {
                    *v37 = _mm_loadu_si128(&v35);
                    v33 = v37;
                  }
                  v21 = (__m128i *)&v33[1];
                  v37 = (__m128i *)&v33[1];
                }
                else
                {
                  v21 = v37;
                  v24 = v44;
                  v23.m128i_i64[1] = v37[-1].m128i_i64[0];
                  v35 = v23;
                  if ( v44 == v45 )
                  {
                    sub_1F07A80(&v43, v44, &v35);
                    v21 = v37;
                  }
                  else
                  {
                    if ( v44 )
                    {
                      *v44 = _mm_loadu_si128(&v35);
                      v24 = v44;
                      v21 = v37;
                    }
                    v44 = (__m128i *)&v24[1];
                  }
                }
              }
            }
            v25 = v21 - 1;
            v37 = (__m128i *)v25;
            if ( v25 == v36 )
            {
              sub_1F05010((__int64)&v39, v22, v23.m128i_i64[1], v8, v13, (_BYTE *)v9);
            }
            else
            {
              v26 = v25[-1].m128i_i64[1];
              sub_1F05010((__int64)&v39, v22, v23.m128i_i64[1], v8, v13, (_BYTE *)v9);
              v9 = v37[-1].m128i_i64[0];
              *(_DWORD *)(*(_QWORD *)(v39 + 8) + 8LL * *(unsigned int *)(v9 + 192)) += *(_DWORD *)(*(_QWORD *)(v39 + 8)
                                                                                                 + 8LL
                                                                                                 * *(unsigned int *)((*(_QWORD *)(v26 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 192));
              v27 = *(_QWORD *)(v26 - 16) & 0xFFFFFFFFFFFFFFF8LL;
              v8 = *(unsigned int *)(v27 + 192);
              v28 = (_DWORD *)(*(_QWORD *)(v39 + 8) + 8 * v8);
              v13 = v8;
              if ( (_DWORD)v8 == v28[1] )
              {
                v29 = *(_BYTE **)(v27 + 112);
                v8 = (__int64)&v29[16 * *(unsigned int *)(v27 + 120)];
                if ( v29 == (_BYTE *)v8 )
                {
LABEL_34:
                  if ( *v28 <= *(_DWORD *)(v39 + 4) )
                  {
                    v31 = *(_DWORD *)(v9 + 192);
                    v28[1] = v31;
                    sub_3945B70(v40, v31, (unsigned int)v13);
                  }
                }
                else
                {
                  v30 = 0;
                  while ( (*v29 & 6) != 0 || (unsigned int)++v30 <= 3 )
                  {
                    v29 += 16;
                    if ( (_BYTE *)v8 == v29 )
                      goto LABEL_34;
                  }
                }
              }
            }
            v21 = v37;
          }
          while ( v37 != v36 );
          if ( v37 )
            j_j___libc_free_0(v37, (char *)v38 - (char *)v37);
        }
        else
        {
          while ( (*v14 & 6) != 0 || *(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 192) == -1 )
          {
            v14 += 2;
            if ( v15 == v14 )
              goto LABEL_11;
          }
        }
      }
      v3 += 272;
    }
    while ( v12 != v3 );
  }
  sub_1F07480(&v39);
  _libc_free(v49);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( v43 )
    j_j___libc_free_0(v43, (char *)v45 - (char *)v43);
  if ( (_BYTE *)v40[0] != v41 )
    _libc_free(v40[0]);
}
