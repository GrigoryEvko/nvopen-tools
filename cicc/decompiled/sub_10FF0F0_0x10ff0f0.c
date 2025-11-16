// Function: sub_10FF0F0
// Address: 0x10ff0f0
//
__int64 __fastcall sub_10FF0F0(unsigned __int8 *a1, __int64 a2, int *a3, const __m128i *a4, __int64 a5)
{
  unsigned int v8; // eax
  unsigned int v9; // r15d
  __int64 v10; // rdx
  unsigned int v12; // r11d
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v14; // rcx
  unsigned __int8 *v15; // rdx
  unsigned __int8 *v16; // rbx
  char v17; // al
  unsigned __int8 *v18; // rcx
  unsigned __int8 *v19; // rdi
  __int64 v20; // rdx
  unsigned __int8 *v21; // r9
  unsigned __int8 *v22; // rdi
  _QWORD *v23; // r13
  unsigned int v24; // r13d
  unsigned int v25; // eax
  char v26; // al
  unsigned __int8 *v27; // rcx
  unsigned __int8 *v28; // rdi
  __int64 v29; // rdx
  unsigned __int8 *v30; // r9
  unsigned __int8 *v31; // rbx
  _QWORD *v32; // rdx
  int v33; // eax
  __int64 v34; // rax
  _BYTE *v35; // rax
  _BYTE *v36; // rax
  unsigned int v37; // eax
  int v38; // r8d
  unsigned int v39; // edx
  unsigned int v40; // esi
  __int64 v41; // rdi
  __int64 v42; // rax
  __m128i v43; // xmm1
  unsigned __int64 v44; // xmm2_8
  __m128i v45; // xmm3
  unsigned int v46; // r13d
  int v47; // [rsp+4h] [rbp-BCh]
  unsigned int v48; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v49; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v50; // [rsp+10h] [rbp-B0h]
  int v51; // [rsp+10h] [rbp-B0h]
  int v53; // [rsp+2Ch] [rbp-94h] BYREF
  __int64 v54; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v55; // [rsp+38h] [rbp-88h]
  __m128i v56[2]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v57; // [rsp+60h] [rbp-60h]
  __int64 v58; // [rsp+68h] [rbp-58h]
  __m128i v59; // [rsp+70h] [rbp-50h]
  __int64 v60; // [rsp+80h] [rbp-40h]

  *a3 = 0;
  v8 = sub_10FD310(a1, a2);
  if ( (_BYTE)v8 )
  {
    return 1;
  }
  else
  {
    v9 = v8;
    if ( *a1 > 0x1Cu )
    {
      v10 = *((_QWORD *)a1 + 2);
      if ( v10 )
      {
        if ( !*(_QWORD *)(v10 + 8) )
        {
          switch ( *a1 )
          {
            case '*':
            case ',':
            case '.':
            case '9':
            case ':':
            case ';':
              if ( (a1[7] & 0x40) != 0 )
                v13 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
              else
                v13 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
              if ( !(unsigned __int8)sub_10FF0F0(*(_QWORD *)v13, a2, a3, a4, a5) )
                return v9;
              v14 = (a1[7] & 0x40) != 0
                  ? (unsigned __int8 *)*((_QWORD *)a1 - 1)
                  : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
              if ( !(unsigned __int8)sub_10FF0F0(*((_QWORD *)v14 + 4), a2, &v53, a4, a5) )
                return v9;
              if ( !*a3 )
              {
                LOBYTE(v9) = v53 == 0;
                return v9;
              }
              if ( v53 || (unsigned int)*a1 - 57 > 2 )
                return v9;
              v51 = *a3;
              v37 = sub_BCB060(*((_QWORD *)a1 + 1));
              v38 = v51;
              v55 = v37;
              v39 = v37;
              if ( v37 > 0x40 )
              {
                sub_C43690((__int64)&v54, 0, 0);
                v39 = v55;
                v38 = v51;
              }
              else
              {
                v54 = 0;
              }
              v40 = v39 - v38;
              if ( v39 - v38 != v39 )
              {
                if ( v40 > 0x3F || v39 > 0x40 )
                  sub_C43C90(&v54, v40, v39);
                else
                  v54 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) << v40;
              }
              v41 = *(_QWORD *)(sub_986520((__int64)a1) + 32);
              v42 = a4[10].m128i_i64[0];
              v43 = _mm_loadu_si128(a4 + 7);
              v56[0] = _mm_loadu_si128(a4 + 6);
              v44 = _mm_loadu_si128(a4 + 8).m128i_u64[0];
              v45 = _mm_loadu_si128(a4 + 9);
              v60 = v42;
              v57 = v44;
              v56[1] = v43;
              v58 = a5;
              v59 = v45;
              v46 = sub_9AC230(v41, (__int64)&v54, v56, 0);
              if ( v55 > 0x40 && v54 )
                j_j___libc_free_0_0(v54);
              if ( !(_BYTE)v46 )
                return v9;
              if ( *a1 != 57 )
                return 1;
              *a3 = 0;
              return v46;
            case '6':
              v26 = a1[7] & 0x40;
              if ( v26 )
                v27 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
              else
                v27 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
              v28 = (unsigned __int8 *)*((_QWORD *)v27 + 4);
              v29 = *v28;
              v30 = v28 + 24;
              if ( (_BYTE)v29 == 17 )
                goto LABEL_42;
              if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v28 + 1) + 8LL) - 17 <= 1
                && (unsigned __int8)v29 <= 0x15u )
              {
                v36 = sub_AD7630((__int64)v28, 0, v29);
                if ( v36 )
                {
                  if ( *v36 == 17 )
                  {
                    v30 = v36 + 24;
                    v26 = a1[7] & 0x40;
LABEL_42:
                    if ( v26 )
                      v31 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
                    else
                      v31 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
                    v50 = v30;
                    v9 = sub_10FF0F0(*(_QWORD *)v31, a2, a3, a4, a5);
                    if ( (_BYTE)v9 )
                    {
                      v32 = *(_QWORD **)v50;
                      if ( *((_DWORD *)v50 + 2) > 0x40u )
                        v32 = (_QWORD *)*v32;
                      v33 = *a3 - (_DWORD)v32;
                      if ( (unsigned int)*a3 <= (unsigned __int64)v32 )
                        v33 = 0;
                      *a3 = v33;
                    }
                  }
                }
              }
              return v9;
            case '7':
              v17 = a1[7] & 0x40;
              if ( v17 )
                v18 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
              else
                v18 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
              v19 = (unsigned __int8 *)*((_QWORD *)v18 + 4);
              v20 = *v19;
              v21 = v19 + 24;
              if ( (_BYTE)v20 == 17 )
                goto LABEL_32;
              if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v19 + 1) + 8LL) - 17 <= 1
                && (unsigned __int8)v20 <= 0x15u )
              {
                v35 = sub_AD7630((__int64)v19, 0, v20);
                if ( v35 )
                {
                  if ( *v35 == 17 )
                  {
                    v21 = v35 + 24;
                    v17 = a1[7] & 0x40;
LABEL_32:
                    if ( v17 )
                      v22 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
                    else
                      v22 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
                    v49 = v21;
                    v9 = sub_10FF0F0(*(_QWORD *)v22, a2, a3, a4, a5);
                    if ( (_BYTE)v9 )
                    {
                      v23 = *(_QWORD **)v49;
                      if ( *((_DWORD *)v49 + 2) > 0x40u )
                        v23 = (_QWORD *)*v23;
                      v24 = *a3 + (_DWORD)v23;
                      *a3 = v24;
                      v25 = sub_BCB060(*((_QWORD *)a1 + 1));
                      if ( v24 > v25 )
                        *a3 = v25;
                    }
                  }
                }
              }
              return v9;
            case 'C':
            case 'D':
            case 'E':
              return 1;
            case 'T':
              if ( !(unsigned __int8)sub_10FF0F0(**((_QWORD **)a1 - 1), a2, a3, a4, a5) )
                return v9;
              v47 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
              if ( v47 == 1 )
                return 1;
              v12 = 1;
              break;
            case 'U':
              v34 = *((_QWORD *)a1 - 4);
              if ( v34
                && !*(_BYTE *)v34
                && *(_QWORD *)(v34 + 24) == *((_QWORD *)a1 + 10)
                && (*(_BYTE *)(v34 + 33) & 0x20) != 0 )
              {
                LOBYTE(v9) = *(_DWORD *)(v34 + 36) == 493;
              }
              return v9;
            case 'V':
              if ( (a1[7] & 0x40) != 0 )
                v15 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
              else
                v15 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
              if ( (unsigned __int8)sub_10FF0F0(*((_QWORD *)v15 + 4), a2, &v53, a4, a5) )
              {
                v16 = (a1[7] & 0x40) != 0
                    ? (unsigned __int8 *)*((_QWORD *)a1 - 1)
                    : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
                if ( (unsigned __int8)sub_10FF0F0(*((_QWORD *)v16 + 8), a2, a3, a4, a5) )
                  LOBYTE(v9) = *a3 == v53;
              }
              return v9;
            default:
              return v9;
          }
          while ( 1 )
          {
            v48 = v12;
            if ( !(unsigned __int8)sub_10FF0F0(*(_QWORD *)(*((_QWORD *)a1 - 1) + 32LL * v12), a2, &v53, a4, a5)
              || *a3 != v53 )
            {
              break;
            }
            v12 = v48 + 1;
            if ( v48 + 1 == v47 )
              return 1;
          }
        }
      }
    }
  }
  return v9;
}
