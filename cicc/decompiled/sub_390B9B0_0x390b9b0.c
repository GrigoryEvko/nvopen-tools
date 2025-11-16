// Function: sub_390B9B0
// Address: 0x390b9b0
//
unsigned __int64 __fastcall sub_390B9B0(__int64 a1, _QWORD *a2, __int64 a3, _QWORD *a4)
{
  __int64 v7; // rbx
  unsigned __int64 result; // rax
  int v9; // ecx
  _BYTE *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int8 v16; // cl
  int v17; // r15d
  __int64 v18; // kr00_8
  size_t v19; // rdx
  __int64 v20; // r15
  char v21; // si
  size_t *v22; // rax
  __int64 v23; // rsi
  unsigned __int64 v24; // rdi
  __m128i *v25; // r10
  __int64 v26; // rdx
  __int64 v27; // rax
  char v28; // cl
  size_t v29; // rcx
  size_t v30; // r15
  size_t v31; // r8
  __int64 v32; // r11
  _QWORD *v33; // rdx
  size_t v34; // rdx
  unsigned int v35; // eax
  unsigned __int32 v36; // edx
  size_t v37; // rsi
  unsigned int v38; // ecx
  __int64 v39; // r11
  unsigned int v40; // r15d
  __int64 v41; // rdx
  __int64 v42; // rax
  __int16 v43; // dx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // rax
  unsigned __int32 v47; // edx
  size_t v48; // [rsp+10h] [rbp-180h]
  __int64 v49; // [rsp+18h] [rbp-178h]
  size_t v50; // [rsp+20h] [rbp-170h]
  __m128i *v51; // [rsp+20h] [rbp-170h]
  __int64 v52; // [rsp+20h] [rbp-170h]
  __int64 v53; // [rsp+20h] [rbp-170h]
  __int64 v54; // [rsp+20h] [rbp-170h]
  __int64 v55; // [rsp+20h] [rbp-170h]
  size_t v57; // [rsp+30h] [rbp-160h] BYREF
  size_t v58; // [rsp+38h] [rbp-158h] BYREF
  __m128i v59; // [rsp+40h] [rbp-150h] BYREF
  char v60; // [rsp+50h] [rbp-140h]
  char v61; // [rsp+51h] [rbp-13Fh]
  __m128i v62; // [rsp+60h] [rbp-130h] BYREF
  __int16 v63; // [rsp+70h] [rbp-120h]
  __m128i v64[2]; // [rsp+80h] [rbp-110h] BYREF
  __m128i v65; // [rsp+A0h] [rbp-F0h] BYREF
  char v66; // [rsp+B0h] [rbp-E0h]
  char v67; // [rsp+B1h] [rbp-DFh]
  __m128i v68; // [rsp+C0h] [rbp-D0h] BYREF
  char v69; // [rsp+D0h] [rbp-C0h]
  char v70; // [rsp+D1h] [rbp-BFh]
  __m128i v71; // [rsp+E0h] [rbp-B0h] BYREF
  __int16 v72; // [rsp+F0h] [rbp-A0h]
  __m128i v73[2]; // [rsp+100h] [rbp-90h] BYREF
  __m128i v74; // [rsp+120h] [rbp-70h] BYREF
  __int16 v75; // [rsp+130h] [rbp-60h]
  __m128i v76; // [rsp+140h] [rbp-50h] BYREF
  __int16 v77; // [rsp+150h] [rbp-40h]

  v7 = a3 + 96;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a3 + 16LL))(a3) )
  {
    for ( result = *(_QWORD *)(a3 + 104); result != v7; result = *(_QWORD *)(result + 8) )
    {
      if ( *(_BYTE *)(result + 16) == 1 )
      {
        if ( 3LL * *(unsigned int *)(result + 120) )
          sub_16BD130("cannot have fixups in virtual section!", 1u);
        v9 = *(_DWORD *)(result + 72);
        if ( v9 )
        {
          v10 = *(_BYTE **)(result + 64);
          v11 = (__int64)&v10[v9 - 1 + 1];
          do
          {
            if ( *v10 )
            {
              if ( *(_DWORD *)(a3 + 144) == 1 )
              {
                v12 = *(_QWORD *)(a3 + 160);
                v13 = *(_QWORD *)(a3 + 152);
                v77 = 770;
                v73[0].m128i_i64[1] = v12;
                v74.m128i_i64[0] = (__int64)"non-zero initializer found in section '";
                v74.m128i_i64[1] = (__int64)v73;
                v73[0].m128i_i64[0] = v13;
                v76.m128i_i64[0] = (__int64)&v74;
                v75 = 1283;
                v76.m128i_i64[1] = (__int64)"'";
                sub_16BCFB0((__int64)&v76, 1u);
              }
              sub_16BD130("non-zero initializer found in virtual section", 1u);
            }
            ++v10;
          }
          while ( v10 != (_BYTE *)v11 );
        }
      }
    }
  }
  else
  {
    result = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 64LL))(a2);
    v14 = *(_QWORD *)(a3 + 104);
    if ( v14 != v7 )
    {
      while ( 1 )
      {
        v15 = sub_390B580(a1, a4, v14);
        v16 = *(_BYTE *)(v14 + 16);
        v57 = v15;
        v17 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL);
        if ( v16 <= 6u && ((1LL << v16) & 0x56) != 0 )
          sub_390B8A0(a1, (__int64)a2, v14, v15);
        v18 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 64LL))(a2);
        result = *(unsigned __int8 *)(v14 + 16);
        switch ( *(_BYTE *)(v14 + 16) )
        {
          case 0:
            v37 = *(unsigned int *)(v14 + 64);
            v38 = *(_DWORD *)(v14 + 64);
            result = v57 / v37;
            v58 = v57 / v37;
            if ( v57 != v57 / v37 * v37 )
            {
              v62.m128i_i32[0] = v37;
              v74.m128i_i64[0] = (__int64)"'";
              v71.m128i_i64[0] = (__int64)&v57;
              v65.m128i_i64[0] = (__int64)"' is not a divisor of padding size '";
              v59.m128i_i64[0] = (__int64)"undefined .align directive, value size '";
              v75 = 259;
              v72 = 267;
              v67 = 1;
              v66 = 3;
              v63 = 265;
              v61 = 1;
              v60 = 3;
              sub_14EC200(v64, &v59, &v62);
              sub_14EC200(&v68, v64, &v65);
              goto LABEL_56;
            }
            if ( (*(_BYTE *)(v14 + 52) & 1) != 0 )
            {
              result = (*(__int64 (__fastcall **)(_QWORD, _QWORD *, unsigned __int64))(**(_QWORD **)(a1 + 8) + 120LL))(
                         *(_QWORD *)(a1 + 8),
                         a2,
                         result);
              if ( !(_BYTE)result )
              {
                v75 = 259;
                v72 = 267;
                v74.m128i_i64[0] = (__int64)" bytes";
                v22 = &v58;
LABEL_55:
                v71.m128i_i64[0] = (__int64)v22;
                v70 = 1;
                v68.m128i_i64[0] = (__int64)"unable to write nop sequence of ";
                v69 = 3;
LABEL_56:
                sub_14EC200(v73, &v68, &v71);
                sub_14EC200(&v76, v73, &v74);
                sub_16BCFB0((__int64)&v76, 1u);
              }
            }
            else if ( v57 >= v37 )
            {
              v39 = 0;
              v40 = v17 - 1;
              while ( 1 )
              {
                if ( v38 == 4 )
                {
                  v46 = *(_QWORD *)(v14 + 56);
                  v54 = v39;
                  v47 = _byteswap_ulong(v46);
                  if ( v40 > 1 )
                    LODWORD(v46) = v47;
                  v76.m128i_i32[0] = v46;
                  result = sub_16E7EE0((__int64)a2, v76.m128i_i8, 4u);
                  v39 = v54;
                }
                else if ( v38 > 4 )
                {
                  v44 = *(_QWORD *)(v14 + 56);
                  v53 = v39;
                  v45 = _byteswap_uint64(v44);
                  if ( v40 > 1 )
                    v44 = v45;
                  v76.m128i_i64[0] = v44;
                  result = sub_16E7EE0((__int64)a2, v76.m128i_i8, 8u);
                  v39 = v53;
                }
                else if ( v38 == 1 )
                {
                  v41 = *(_QWORD *)(v14 + 56);
                  result = a2[3];
                  if ( result >= a2[2] )
                  {
                    v55 = v39;
                    result = sub_16E7DE0((__int64)a2, v41);
                    v39 = v55;
                  }
                  else
                  {
                    a2[3] = result + 1;
                    *(_BYTE *)result = v41;
                  }
                }
                else
                {
                  v42 = *(_QWORD *)(v14 + 56);
                  v52 = v39;
                  v43 = __ROL2__(v42, 8);
                  if ( v40 > 1 )
                    LOWORD(v42) = v43;
                  v76.m128i_i16[0] = v42;
                  result = sub_16E7EE0((__int64)a2, v76.m128i_i8, 2u);
                  v39 = v52;
                }
                if ( v58 == ++v39 )
                  break;
                v38 = *(_DWORD *)(v14 + 64);
              }
            }
LABEL_14:
            v14 = *(_QWORD *)(v14 + 8);
            if ( v14 == v7 )
              return result;
            break;
          case 1:
          case 2:
          case 4:
          case 6:
          case 8:
          case 0xC:
            result = sub_16E7EE0((__int64)a2, *(char **)(v14 + 64), *(unsigned int *)(v14 + 72));
            goto LABEL_14;
          case 3:
            v23 = *(unsigned __int8 *)(v14 + 56);
            v24 = *(_QWORD *)(v14 + 48);
            v25 = &v76;
            v26 = v23;
            if ( !*(_BYTE *)(v14 + 56) )
              goto LABEL_89;
            v27 = 0;
            v25 = &v76;
            do
            {
              v28 = v23 - 1 - v27;
              if ( v17 == 1 )
                v28 = v27;
              v76.m128i_i8[v27++] = v24 >> (8 * v28);
            }
            while ( (_DWORD)v23 != (_DWORD)v27 );
            if ( (_DWORD)v23 != 16 )
            {
LABEL_89:
              do
              {
                v76.m128i_i8[v26] = v76.m128i_i8[v26 - v23];
                ++v26;
              }
              while ( (unsigned int)v26 <= 0xF );
            }
            v29 = v57;
            v30 = (unsigned int)v23 * (0x10 / (unsigned int)v23);
            v31 = v57 / v30;
            if ( v30 > v57 )
              goto LABEL_44;
            v32 = 0;
            do
            {
              while ( 1 )
              {
                v33 = (_QWORD *)a2[3];
                if ( v30 <= a2[2] - (_QWORD)v33 )
                  break;
                v48 = v31;
                v49 = v32;
                v51 = v25;
                sub_16E7EE0((__int64)a2, v25->m128i_i8, v30);
                v31 = v48;
                v25 = v51;
                v32 = v49 + 1;
                if ( v48 == v49 + 1 )
                  goto LABEL_43;
              }
              if ( (_DWORD)v23 * (0x10 / (unsigned int)v23) )
              {
                if ( (unsigned int)v30 >= 8 )
                {
                  *v33 = v25->m128i_i64[0];
                  *(_QWORD *)((char *)v33 + v30 - 8) = *(__int64 *)((char *)&v25->m128i_i64[-1] + v30);
                  qmemcpy(
                    (void *)((unsigned __int64)(v33 + 1) & 0xFFFFFFFFFFFFFFF8LL),
                    (const void *)((char *)v25 - ((char *)v33 - ((unsigned __int64)(v33 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
                    8LL * (((unsigned int)v30 + (_DWORD)v33 - (((_DWORD)v33 + 8) & 0xFFFFFFF8)) >> 3));
                }
                else if ( (v30 & 4) != 0 )
                {
                  *(_DWORD *)v33 = v25->m128i_i32[0];
                  *(_DWORD *)((char *)v33 + v30 - 4) = *(__int32 *)((char *)&v25->m128i_i32[-1] + v30);
                }
                else
                {
                  *(_BYTE *)v33 = v25->m128i_i8[0];
                  if ( (v30 & 2) != 0 )
                    *(_WORD *)((char *)v33 + v30 - 2) = *(__int16 *)((char *)&v25->m128i_i16[-1] + v30);
                }
                a2[3] += v30;
              }
              ++v32;
            }
            while ( v31 != v32 );
LABEL_43:
            v29 = v57;
LABEL_44:
            result = v29 / v30;
            v34 = v29 % v30;
            if ( v29 % v30 )
LABEL_49:
              result = sub_16E7EE0((__int64)a2, v25->m128i_i8, v34);
            goto LABEL_14;
          case 5:
            v19 = v57;
            if ( v57 )
            {
              v20 = 0;
              do
              {
                v21 = *(_BYTE *)(v14 + 56);
                result = a2[3];
                if ( result < a2[2] )
                {
                  a2[3] = result + 1;
                  *(_BYTE *)result = v21;
                }
                else
                {
                  v50 = v19;
                  result = sub_16E7DE0((__int64)a2, v21);
                  v19 = v50;
                }
                ++v20;
              }
              while ( v19 != v20 );
            }
            goto LABEL_14;
          case 7:
            result = sub_16E7EE0((__int64)a2, *(char **)(v14 + 56), *(unsigned int *)(v14 + 64));
            goto LABEL_14;
          case 9:
            result = (*(__int64 (__fastcall **)(_QWORD, _QWORD *, size_t))(**(_QWORD **)(a1 + 8) + 120LL))(
                       *(_QWORD *)(a1 + 8),
                       a2,
                       v57);
            if ( (_BYTE)result )
              goto LABEL_14;
            v74.m128i_i64[0] = (__int64)" bytes";
            v22 = &v57;
            v75 = 259;
            v72 = 267;
            goto LABEL_55;
          case 0xA:
            v25 = &v76;
            v35 = *(_DWORD *)(*(_QWORD *)(v14 + 48) + 16LL);
            v36 = _byteswap_ulong(v35);
            if ( (unsigned int)(v17 - 1) > 1 )
              v35 = v36;
            v34 = 4;
            v76.m128i_i32[0] = v35;
            goto LABEL_49;
          case 0xB:
            result = sub_16E7EE0((__int64)a2, *(char **)(v14 + 80), *(unsigned int *)(v14 + 88));
            goto LABEL_14;
          default:
            result = v18;
            goto LABEL_14;
        }
      }
    }
  }
  return result;
}
