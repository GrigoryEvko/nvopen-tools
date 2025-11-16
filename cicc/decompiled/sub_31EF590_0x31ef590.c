// Function: sub_31EF590
// Address: 0x31ef590
//
void __fastcall sub_31EF590(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // r12
  int v7; // r13d
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 (__fastcall *v13)(__int64, __int64, _QWORD, __int64, unsigned __int8 *); // r14
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // r9
  int v17; // esi
  __int64 v18; // r8
  __int64 v19; // rcx
  int v20; // eax
  __m128i *v21; // rbx
  __int64 v22; // rcx
  __m128i *v23; // rdx
  unsigned __int64 v24; // r10
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r12
  __m128i *v29; // rbx
  __m128i *v30; // r12
  unsigned __int64 v31; // rdi
  __m128i *v32; // r8
  __int64 v33; // r14
  __int64 v34; // rbx
  unsigned int v35; // r13d
  __int64 v36; // rax
  _QWORD *v37; // r14
  void *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rsi
  _QWORD **v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __m128i *v45; // r10
  __m128i *v46; // r13
  __m128i *v47; // r12
  __int64 v48; // rbx
  __int8 v49; // cl
  __int64 v50; // rdx
  char **v51; // rsi
  __int64 v52; // rdi
  __m128i *v53; // r14
  __m128i *v54; // rbx
  unsigned __int64 v55; // rdi
  int v56; // eax
  unsigned __int64 v57; // [rsp+8h] [rbp-198h]
  __int64 v58; // [rsp+10h] [rbp-190h]
  __int64 v59; // [rsp+18h] [rbp-188h]
  int v60; // [rsp+20h] [rbp-180h]
  int v61; // [rsp+20h] [rbp-180h]
  __int64 v62; // [rsp+30h] [rbp-170h]
  __int64 v63; // [rsp+38h] [rbp-168h]
  __int64 v64; // [rsp+40h] [rbp-160h]
  __int64 v65; // [rsp+48h] [rbp-158h]
  __int64 v66; // [rsp+48h] [rbp-158h]
  __int64 v67; // [rsp+48h] [rbp-158h]
  unsigned int v68; // [rsp+50h] [rbp-150h]
  __int64 v69; // [rsp+50h] [rbp-150h]
  __int64 v70; // [rsp+50h] [rbp-150h]
  __int64 v71; // [rsp+50h] [rbp-150h]
  __int64 v72; // [rsp+58h] [rbp-148h]
  unsigned int v73; // [rsp+58h] [rbp-148h]
  char v74; // [rsp+58h] [rbp-148h]
  unsigned __int8 v75; // [rsp+67h] [rbp-139h] BYREF
  unsigned __int64 v76; // [rsp+68h] [rbp-138h] BYREF
  __m128i v77; // [rsp+70h] [rbp-130h] BYREF
  _BYTE *v78; // [rsp+80h] [rbp-120h]
  __int64 v79; // [rsp+88h] [rbp-118h]
  _BYTE v80[16]; // [rsp+90h] [rbp-110h] BYREF
  __m128i *v81; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-F8h]
  _BYTE v83[240]; // [rsp+B0h] [rbp-F0h] BYREF

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 56LL);
  v63 = v1;
  if ( *(_QWORD *)(v1 + 8) != *(_QWORD *)(v1 + 16) )
  {
    v81 = (__m128i *)v83;
    v82 = 0x400000000LL;
    v2 = v1;
    v3 = *(_QWORD *)(v1 + 8);
    v4 = (*(_QWORD *)(v2 + 16) - v3) >> 4;
    if ( (_DWORD)v4 )
    {
      v6 = 0;
      v65 = (unsigned int)v4;
      while ( 1 )
      {
        v7 = v6;
        v8 = v3 + 16 * v6;
        v75 = *(_BYTE *)(v8 + 8);
        v9 = sub_31DA930(a1);
        LOBYTE(v10) = sub_2E7A190(v8, v9);
        v11 = 0;
        if ( !*(_BYTE *)(v8 + 9) )
          v11 = *(_QWORD *)v8;
        v68 = v10;
        v72 = v11;
        v12 = sub_31DA6B0(a1);
        v13 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, unsigned __int8 *))(*(_QWORD *)v12 + 64LL);
        v14 = sub_31DA930(a1);
        v15 = v13(v12, v14, v68, v72, &v75);
        v17 = v82;
        v18 = (__int64)v81;
        v19 = v15;
        v20 = v82;
        while ( v20 )
        {
          v21 = &v81[3 * (unsigned int)--v20];
          if ( v21->m128i_i64[0] == v19 )
            goto LABEL_16;
        }
        v77.m128i_i64[0] = v19;
        v16 = v80;
        v22 = HIDWORD(v82);
        v78 = v80;
        v77.m128i_i8[8] = v75;
        v23 = &v77;
        v79 = 0x400000000LL;
        v24 = (unsigned int)v82 + 1LL;
        v25 = 48LL * (unsigned int)v82;
        v26 = v25;
        if ( v24 > HIDWORD(v82) )
        {
          if ( v81 > &v77 || &v77 >= (__m128i *)&v81->m128i_i8[v25] )
          {
            v57 = -1;
            v74 = 0;
          }
          else
          {
            v74 = 1;
            v57 = 0xAAAAAAAAAAAAAAABLL * (&v77 - v81);
          }
          v44 = sub_C8D7D0((__int64)&v81, (__int64)v83, v24, 0x30u, &v76, (__int64)v80);
          v16 = v80;
          v18 = v44;
          v45 = &v81[3 * (unsigned int)v82];
          if ( v81 != v45 )
          {
            v60 = v6;
            v46 = v81;
            v59 = v6;
            v47 = &v81[3 * (unsigned int)v82];
            v58 = v25;
            v48 = v44;
            v70 = v44;
            do
            {
              while ( 1 )
              {
                if ( v48 )
                {
                  *(_QWORD *)v48 = v46->m128i_i64[0];
                  v49 = v46->m128i_i8[8];
                  *(_DWORD *)(v48 + 24) = 0;
                  *(_BYTE *)(v48 + 8) = v49;
                  v22 = v48 + 32;
                  *(_QWORD *)(v48 + 16) = v48 + 32;
                  *(_DWORD *)(v48 + 28) = 4;
                  v50 = v46[1].m128i_u32[2];
                  if ( (_DWORD)v50 )
                    break;
                }
                v46 += 3;
                v48 += 48;
                if ( v47 == v46 )
                  goto LABEL_55;
              }
              v51 = (char **)&v46[1];
              v52 = v48 + 16;
              v46 += 3;
              v48 += 48;
              sub_31D52E0(v52, v51, v50, v22, v18, (__int64)v16);
            }
            while ( v47 != v46 );
LABEL_55:
            v16 = v80;
            v18 = v70;
            v7 = v60;
            v6 = v59;
            v25 = v58;
            v45 = &v81[3 * (unsigned int)v82];
            if ( v81 != v45 )
            {
              v53 = v81;
              v54 = &v81[3 * (unsigned int)v82];
              do
              {
                v54 -= 3;
                v55 = v54[1].m128i_u64[0];
                if ( (__m128i *)v55 != &v54[2] )
                  _libc_free(v55);
              }
              while ( v53 != v54 );
              v18 = v70;
              v25 = v58;
              v16 = v80;
              v45 = v81;
            }
          }
          v56 = v76;
          if ( v45 != (__m128i *)v83 )
          {
            v61 = v76;
            v71 = v18;
            _libc_free((unsigned __int64)v45);
            v16 = v80;
            v56 = v61;
            v18 = v71;
          }
          HIDWORD(v82) = v56;
          v81 = (__m128i *)v18;
          v23 = &v77;
          v17 = v82;
          v26 = 48LL * (unsigned int)v82;
          if ( v74 )
          {
            v22 = v57;
            v23 = (__m128i *)(v18 + 48 * v57);
          }
        }
        v18 += v26;
        if ( v18 )
        {
          *(_QWORD *)v18 = v23->m128i_i64[0];
          *(_BYTE *)(v18 + 8) = v23->m128i_i8[8];
          *(_QWORD *)(v18 + 16) = v18 + 32;
          *(_QWORD *)(v18 + 24) = 0x400000000LL;
          if ( v23[1].m128i_i32[2] )
          {
            sub_31D52E0(v18 + 16, (char **)&v23[1], (__int64)v23, v22, v18, (__int64)v80);
            v17 = v82;
            v16 = v80;
          }
          else
          {
            v17 = v82;
          }
        }
        LODWORD(v82) = v17 + 1;
        if ( v78 != v80 )
          _libc_free((unsigned __int64)v78);
        v21 = (__m128i *)((char *)v81 + v25);
LABEL_16:
        if ( (unsigned int)v21->m128i_i8[8] < v75 )
          v21->m128i_i8[8] = v75;
        v27 = v21[1].m128i_u32[2];
        if ( v27 + 1 > (unsigned __int64)v21[1].m128i_u32[3] )
        {
          sub_C8D5F0((__int64)v21[1].m128i_i64, &v21[2], v27 + 1, 4u, v18, (__int64)v16);
          v27 = v21[1].m128i_u32[2];
        }
        ++v6;
        *(_DWORD *)(v21[1].m128i_i64[0] + 4 * v27) = v7;
        ++v21[1].m128i_i32[2];
        if ( v65 == v6 )
          break;
        v3 = *(_QWORD *)(v63 + 8);
      }
      v32 = v81;
      if ( (_DWORD)v82 )
      {
        v28 = 0;
        v73 = 0;
        v64 = 0;
        v62 = 48LL * (unsigned int)v82;
        do
        {
          v33 = v32[v28 / 0x10 + 1].m128i_u32[2];
          if ( (_DWORD)v33 )
          {
            v34 = 0;
            v69 = 4 * v33;
            do
            {
              v35 = *(_DWORD *)(v32[v28 / 0x10 + 1].m128i_i64[0] + v34);
              v36 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 320LL))(a1, v35);
              v37 = (_QWORD *)v36;
              if ( !*(_QWORD *)v36 )
              {
                if ( (*(_BYTE *)(v36 + 9) & 0x70) != 0x20
                  || *(char *)(v36 + 8) < 0
                  || (*(_BYTE *)(v36 + 8) |= 8u, v38 = sub_E807D0(*(_QWORD *)(v36 + 24)), (*v37 = v38) == 0) )
                {
                  v39 = v73;
                  v40 = v81[v28 / 0x10].m128i_i64[0];
                  if ( v40 != v64 )
                  {
                    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
                      *(_QWORD *)(a1 + 224),
                      v40,
                      0);
                    sub_31DCA70(a1, v81[v28 / 0x10].m128i_u8[8], 0, 0);
                    v39 = 0;
                    v73 = 0;
                    v64 = v81[v28 / 0x10].m128i_i64[0];
                  }
                  v41 = *(_QWORD ***)(a1 + 224);
                  v77 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v63 + 8) + 16LL * v35));
                  v66 = -(1LL << v77.m128i_i8[8]) & ((1LL << v77.m128i_i8[8]) + v39 - 1);
                  sub_E99300(v41, (unsigned int)v66 - v73);
                  v42 = sub_31DA930(a1);
                  v73 = v66 + sub_2E7A0A0((__int64)&v77, v42);
                  (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(
                    *(_QWORD *)(a1 + 224),
                    v37,
                    0);
                  if ( v77.m128i_i8[9] )
                  {
                    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 344LL))(a1, v77.m128i_i64[0]);
                  }
                  else
                  {
                    v67 = v77.m128i_i64[0];
                    v43 = sub_31DA930(a1);
                    sub_31EA6F0(a1, v43, v67, 0);
                  }
                }
              }
              v32 = v81;
              v34 += 4;
            }
            while ( v69 != v34 );
          }
          v28 += 48LL;
        }
        while ( v62 != v28 );
        v29 = &v32[3 * (unsigned int)v82];
        if ( v29 != v32 )
        {
          v30 = v32;
          do
          {
            v29 -= 3;
            v31 = v29[1].m128i_u64[0];
            if ( (__m128i *)v31 != &v29[2] )
              _libc_free(v31);
          }
          while ( v29 != v30 );
          v32 = v81;
        }
      }
      if ( v32 != (__m128i *)v83 )
        _libc_free((unsigned __int64)v32);
    }
  }
}
