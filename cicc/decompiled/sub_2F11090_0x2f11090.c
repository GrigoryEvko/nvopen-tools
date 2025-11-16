// Function: sub_2F11090
// Address: 0x2f11090
//
void __fastcall sub_2F11090(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        _DWORD *a4,
        __int64 a5,
        char a6,
        __int64 a7,
        char a8)
{
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned int v15; // edx
  __int64 *v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // r12
  const char *v19; // rdi
  size_t v20; // rax
  unsigned __int8 *v21; // rdi
  __int16 v22; // ax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // r12
  size_t v26; // rdx
  __m128i *v27; // rsi
  __m128i *v28; // rax
  void *v29; // rdx
  char v30; // dl
  unsigned int v31; // ebx
  __int64 v32; // rdx
  int v33; // eax
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  int v36; // r8d
  int v38; // [rsp+Ch] [rbp-B4h]
  _BYTE *v39[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+20h] [rbp-A0h] BYREF
  __m128i *v41; // [rsp+30h] [rbp-90h] BYREF
  size_t v42; // [rsp+38h] [rbp-88h]
  __m128i v43; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v44; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+58h] [rbp-68h]
  __m128i v46; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int8 *v47; // [rsp+70h] [rbp-50h] BYREF
  size_t v48; // [rsp+78h] [rbp-48h]
  _QWORD v49[8]; // [rsp+80h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  (*(void (__fastcall **)(_BYTE **, __int64, __int64, __int64, _QWORD, _DWORD *))(*(_QWORD *)a5 + 936LL))(
    v39,
    a5,
    a2,
    v10,
    a3,
    a4);
  switch ( *(_BYTE *)v10 )
  {
    case 0:
    case 2:
    case 3:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xD:
    case 0xE:
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
      v23 = 0;
      if ( !*(_BYTE *)v10 && a6 && (*(_WORD *)(v10 + 2) & 0xFF0) != 0 && (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
        v23 = sub_2E89F40(*(_QWORD *)(v10 + 16), a3);
      goto LABEL_17;
    case 1:
      v22 = *(_WORD *)(a2 + 68);
      if ( v22 == 8 )
      {
        if ( a3 != 2 )
          goto LABEL_16;
LABEL_31:
        sub_2EABA60(*a1, v10);
        sub_2EAB970(*a1, *(_QWORD *)(v10 + 24), (__int64)a4);
        goto LABEL_3;
      }
      if ( v22 != 9 )
      {
        if ( v22 == 19 )
        {
          if ( a3 <= 1 || (a3 & 1) != 0 )
            goto LABEL_16;
        }
        else if ( v22 != 12 || a3 != 3 )
        {
          goto LABEL_16;
        }
        goto LABEL_31;
      }
      if ( a3 == 3 )
        goto LABEL_31;
LABEL_16:
      v23 = 0;
LABEL_17:
      v24 = a1[1];
      LODWORD(v47) = a3;
      BYTE4(v47) = 1;
      sub_2EAE5A0(v10, *a1, v24, a7, (__int64)v47, a8, 0, a6, v23, (__int64)a4);
      v25 = *a1;
      v41 = &v43;
      sub_2F07250((__int64 *)&v41, v39[0], (__int64)&v39[0][(unsigned __int64)v39[1]]);
      v26 = v42;
      if ( v42 )
      {
        v48 = 0;
        v47 = (unsigned __int8 *)v49;
        LOBYTE(v49[0]) = 0;
        sub_2240E30((__int64)&v47, v42 + 4);
        if ( 0x3FFFFFFFFFFFFFFFLL - v48 <= 3
          || (sub_2241490((unsigned __int64 *)&v47, " /* ", 4u),
              sub_2241490((unsigned __int64 *)&v47, v41->m128i_i8, v42),
              0x3FFFFFFFFFFFFFFFLL - v48 <= 2) )
        {
          sub_4262D8((__int64)"basic_string::append");
        }
        v28 = (__m128i *)sub_2241490((unsigned __int64 *)&v47, " */", 3u);
        v44 = &v46;
        if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
        {
          v46 = _mm_loadu_si128(v28 + 1);
        }
        else
        {
          v44 = (__m128i *)v28->m128i_i64[0];
          v46.m128i_i64[0] = v28[1].m128i_i64[0];
        }
        v45 = v28->m128i_i64[1];
        v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
        v28->m128i_i64[1] = 0;
        v28[1].m128i_i8[0] = 0;
        if ( v47 != (unsigned __int8 *)v49 )
          j_j___libc_free_0((unsigned __int64)v47);
        v26 = v45;
        v27 = v44;
      }
      else
      {
        v27 = v41;
        v44 = &v46;
        if ( v41 == &v43 )
        {
          v27 = &v46;
          v46 = _mm_load_si128(&v43);
        }
        else
        {
          v44 = v41;
          v46.m128i_i64[0] = v43.m128i_i64[0];
        }
        v45 = 0;
        v41 = &v43;
        v42 = 0;
        v43.m128i_i8[0] = 0;
      }
      sub_CB6200(v25, (unsigned __int8 *)v27, v26);
      if ( v44 != &v46 )
        j_j___libc_free_0((unsigned __int64)v44);
      v21 = (unsigned __int8 *)v41;
      if ( v41 != &v43 )
LABEL_12:
        j_j___libc_free_0((unsigned __int64)v21);
LABEL_3:
      if ( (__int64 *)v39[0] != &v40 )
        j_j___libc_free_0((unsigned __int64)v39[0]);
      return;
    case 5:
      sub_2F11000(a1, *(_DWORD *)(v10 + 24));
      goto LABEL_3;
    case 0xC:
      v11 = a1[2];
      v12 = *(_QWORD *)(v10 + 24);
      v13 = *(_QWORD *)(v11 + 8);
      v14 = *(unsigned int *)(v11 + 24);
      if ( !(_DWORD)v14 )
        goto LABEL_44;
      v15 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v12 == *v16 )
        goto LABEL_8;
      v36 = 1;
      while ( 2 )
      {
        if ( v17 == -4096 )
        {
LABEL_44:
          v18 = *a1;
        }
        else
        {
          v15 = (v14 - 1) & (v36 + v15);
          v16 = (__int64 *)(v13 + 16LL * v15);
          v17 = *v16;
          if ( v12 != *v16 )
          {
            ++v36;
            continue;
          }
LABEL_8:
          v18 = *a1;
          if ( v16 != (__int64 *)(v13 + 16 * v14) )
          {
            v19 = *(const char **)((*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a4 + 128LL))(a4)
                                 + 8LL * *((unsigned int *)v16 + 2));
            v20 = 0;
            v44 = (__m128i *)v19;
            if ( v19 )
              v20 = strlen(v19);
            v45 = v20;
            sub_C93130((__int64 *)&v47, (__int64)&v44);
            sub_CB6200(v18, v47, v48);
            v21 = v47;
            if ( v47 == (unsigned __int8 *)v49 )
              goto LABEL_3;
            goto LABEL_12;
          }
        }
        break;
      }
      v29 = *(void **)(v18 + 32);
      if ( *(_QWORD *)(v18 + 24) - (_QWORD)v29 <= 0xDu )
      {
        sub_CB6200(v18, "CustomRegMask(", 0xEu);
      }
      else
      {
        qmemcpy(v29, "CustomRegMask(", 14);
        *(_QWORD *)(v18 + 32) += 14LL;
      }
      v30 = 0;
      v31 = 0;
      v38 = a4[4];
      if ( v38 > 0 )
      {
        do
        {
          v33 = *(_DWORD *)(v12 + 4LL * ((int)v31 >> 5));
          if ( _bittest(&v33, v31) )
          {
            if ( v30 )
            {
              v34 = *(_BYTE **)(v18 + 32);
              if ( (unsigned __int64)v34 >= *(_QWORD *)(v18 + 24) )
              {
                sub_CB5D20(v18, 44);
              }
              else
              {
                *(_QWORD *)(v18 + 32) = v34 + 1;
                *v34 = 44;
              }
            }
            sub_2FF6320(&v47, v31, a4, 0, 0);
            if ( !v49[0] )
              sub_4263D6(&v47, v31, v32);
            ((void (__fastcall *)(unsigned __int8 **, __int64))v49[1])(&v47, v18);
            if ( v49[0] )
              ((void (__fastcall *)(unsigned __int8 **, unsigned __int8 **, __int64))v49[0])(&v47, &v47, 3);
            v30 = 1;
          }
          ++v31;
        }
        while ( v38 != v31 );
      }
      v35 = *(_BYTE **)(v18 + 32);
      if ( (unsigned __int64)v35 >= *(_QWORD *)(v18 + 24) )
      {
        sub_CB5D20(v18, 41);
      }
      else
      {
        *(_QWORD *)(v18 + 32) = v35 + 1;
        *v35 = 41;
      }
      goto LABEL_3;
    default:
      goto LABEL_3;
  }
}
