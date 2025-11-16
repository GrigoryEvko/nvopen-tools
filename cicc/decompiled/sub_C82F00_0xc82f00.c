// Function: sub_C82F00
// Address: 0xc82f00
//
__int64 __fastcall sub_C82F00(__int64 a1, _QWORD *a2, size_t a3, char a4)
{
  _QWORD *v4; // r8
  size_t v8; // rax
  char *v9; // rdi
  DIR *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  _BYTE *v14; // rdi
  __int64 v15; // rdx
  size_t v16; // rcx
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  char v20; // al
  _QWORD *v21; // rdi
  unsigned int v22; // ebx
  char *v24; // rdi
  size_t v25; // rdx
  char *v26; // [rsp+10h] [rbp-1C0h] BYREF
  char v27; // [rsp+30h] [rbp-1A0h]
  char v28; // [rsp+31h] [rbp-19Fh]
  char v29[32]; // [rsp+40h] [rbp-190h] BYREF
  __int16 v30; // [rsp+60h] [rbp-170h]
  _QWORD v31[4]; // [rsp+70h] [rbp-160h] BYREF
  __int16 v32; // [rsp+90h] [rbp-140h]
  _QWORD *v33; // [rsp+A0h] [rbp-130h] BYREF
  size_t n; // [rsp+A8h] [rbp-128h]
  _QWORD src[2]; // [rsp+B0h] [rbp-120h] BYREF
  int v36; // [rsp+C0h] [rbp-110h]
  char v37; // [rsp+C4h] [rbp-10Ch]
  __m128i v38; // [rsp+C8h] [rbp-108h] BYREF
  __m128i v39; // [rsp+D8h] [rbp-F8h] BYREF
  __m128i v40; // [rsp+E8h] [rbp-E8h] BYREF
  char *name; // [rsp+100h] [rbp-D0h] BYREF
  size_t v42; // [rsp+108h] [rbp-C8h]
  unsigned __int64 v43; // [rsp+110h] [rbp-C0h]
  _BYTE dest[184]; // [rsp+118h] [rbp-B8h] BYREF

  v4 = a2;
  name = dest;
  v42 = 0;
  v43 = 128;
  if ( a3 > 0x80 )
  {
    sub_C8D290(&name, dest, a3, 1);
    v4 = a2;
    v24 = &name[v42];
  }
  else
  {
    v8 = a3;
    if ( !a3 )
      goto LABEL_3;
    v24 = dest;
  }
  a2 = v4;
  memcpy(v24, v4, a3);
  v8 = a3 + v42;
  v42 = v8;
  if ( v8 + 1 > v43 )
  {
    a2 = dest;
    sub_C8D290(&name, dest, v8 + 1, 1);
    v8 = v42;
  }
LABEL_3:
  name[v8] = 0;
  v9 = name;
  v10 = opendir(name);
  if ( !v10 )
  {
    sub_2241E50(v9, a2, v11, v12, v13);
    v22 = *__errno_location();
    goto LABEL_11;
  }
  *(_QWORD *)a1 = v10;
  LOWORD(v36) = 257;
  v32 = 257;
  v30 = 257;
  v26 = ".";
  v28 = 1;
  v27 = 3;
  sub_C81B70(&name, (__int64)&v26, (__int64)v29, (__int64)v31, (__int64)&v33);
  v32 = 261;
  v31[0] = name;
  a2 = v31;
  v31[1] = v42;
  sub_CA0F50(&v33, v31);
  v37 = a4;
  v14 = *(_BYTE **)(a1 + 8);
  v40.m128i_i64[1] = 0xFFFF00000000LL;
  v36 = 9;
  v38 = 0u;
  v39 = 0u;
  v40.m128i_i64[0] = 0;
  if ( v33 == src )
  {
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *v14 = src[0];
      }
      else
      {
        a2 = src;
        memcpy(v14, src, n);
      }
      v25 = n;
      v14 = *(_BYTE **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 16) = v25;
    v14[v25] = 0;
    v14 = v33;
    goto LABEL_8;
  }
  a2 = (_QWORD *)(a1 + 24);
  v15 = src[0];
  v16 = n;
  if ( v14 == (_BYTE *)(a1 + 24) )
  {
    *(_QWORD *)(a1 + 8) = v33;
    *(_QWORD *)(a1 + 16) = v16;
    *(_QWORD *)(a1 + 24) = v15;
    goto LABEL_25;
  }
  a2 = *(_QWORD **)(a1 + 24);
  *(_QWORD *)(a1 + 8) = v33;
  *(_QWORD *)(a1 + 16) = v16;
  *(_QWORD *)(a1 + 24) = v15;
  if ( !v14 )
  {
LABEL_25:
    v33 = src;
    v14 = src;
    goto LABEL_8;
  }
  v33 = v14;
  src[0] = a2;
LABEL_8:
  n = 0;
  *v14 = 0;
  v17 = _mm_loadu_si128(&v38);
  v18 = _mm_loadu_si128(&v39);
  *(_DWORD *)(a1 + 40) = v36;
  v19 = _mm_loadu_si128(&v40);
  v20 = v37;
  v21 = v33;
  *(__m128i *)(a1 + 48) = v17;
  *(__m128i *)(a1 + 64) = v18;
  *(_BYTE *)(a1 + 44) = v20;
  *(__m128i *)(a1 + 80) = v19;
  if ( v21 != src )
  {
    a2 = (_QWORD *)(src[0] + 1LL);
    j_j___libc_free_0(v21, src[0] + 1LL);
  }
  v22 = sub_C82D80((DIR **)a1, (__int64)a2);
LABEL_11:
  if ( name != dest )
    _libc_free(name, a2);
  return v22;
}
