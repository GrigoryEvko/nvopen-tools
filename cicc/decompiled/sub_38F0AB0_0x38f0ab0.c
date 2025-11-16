// Function: sub_38F0AB0
// Address: 0x38f0ab0
//
__int64 __fastcall sub_38F0AB0(__int64 a1, char a2)
{
  char *v3; // rsi
  bool v4; // zf
  unsigned int v5; // r9d
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rdx
  int v10; // ecx
  unsigned __int64 v11; // r13
  __m128i v12; // xmm0
  bool v13; // cc
  unsigned __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  size_t v23; // rbx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  size_t v26; // rax
  void *v27; // r13
  size_t v28; // r14
  size_t v29; // rbx
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  size_t v32; // rax
  size_t v33; // rax
  char v34; // al
  int v35; // eax
  bool v36; // dl
  __int64 v37; // [rsp+0h] [rbp-90h]
  char v39; // [rsp+Fh] [rbp-81h]
  __int64 v40; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v43; // [rsp+28h] [rbp-68h]
  void *s2; // [rsp+30h] [rbp-60h] BYREF
  size_t n; // [rsp+38h] [rbp-58h]
  char v46; // [rsp+40h] [rbp-50h]
  char v47; // [rsp+41h] [rbp-4Fh]
  unsigned __int64 v48; // [rsp+48h] [rbp-48h]
  unsigned int v49; // [rsp+50h] [rbp-40h]

  v39 = a2;
  v3 = *(char **)(a1 + 400);
  if ( v3 == *(char **)(a1 + 408) )
  {
    sub_38E9AD0((unsigned __int64 *)(a1 + 392), v3, (_QWORD *)(a1 + 380));
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *(_QWORD *)(a1 + 380);
      v3 = *(char **)(a1 + 400);
    }
    *(_QWORD *)(a1 + 400) = v3 + 8;
  }
  v4 = *(_BYTE *)(a1 + 385) == 0;
  *(_DWORD *)(a1 + 380) = 1;
  if ( v4 )
  {
    v7 = sub_3909460(a1);
    v37 = sub_39092A0(v7);
    while ( 1 )
    {
      v8 = *(_QWORD *)(a1 + 152);
      if ( (*(_DWORD *)v8 & 0xFFFFFFEF) == 9 || !*(_DWORD *)v8 )
        break;
      v9 = *(unsigned int *)(a1 + 160);
      *(_BYTE *)(a1 + 258) = 0;
      v10 = v9;
      v11 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v9 - 40) >> 3);
      if ( (unsigned __int64)(40 * v9) > 0x28 )
      {
        do
        {
          v12 = _mm_loadu_si128((const __m128i *)(v8 + 48));
          v13 = *(_DWORD *)(v8 + 32) <= 0x40u;
          *(_DWORD *)v8 = *(_DWORD *)(v8 + 40);
          *(__m128i *)(v8 + 8) = v12;
          if ( !v13 )
          {
            v14 = *(_QWORD *)(v8 + 24);
            if ( v14 )
              j_j___libc_free_0_0(v14);
          }
          v15 = *(_QWORD *)(v8 + 64);
          v8 += 40;
          *(_QWORD *)(v8 - 16) = v15;
          LODWORD(v15) = *(_DWORD *)(v8 + 32);
          *(_DWORD *)(v8 + 32) = 0;
          *(_DWORD *)(v8 - 8) = v15;
          --v11;
        }
        while ( v11 );
        v10 = *(_DWORD *)(a1 + 160);
        v8 = *(_QWORD *)(a1 + 152);
      }
      v16 = (unsigned int)(v10 - 1);
      *(_DWORD *)(a1 + 160) = v16;
      v17 = v8 + 40 * v16;
      if ( *(_DWORD *)(v17 + 32) > 0x40u )
      {
        v18 = *(_QWORD *)(v17 + 24);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      if ( !*(_DWORD *)(a1 + 160) )
      {
        sub_392C2E0(&s2, a1 + 144);
        sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)&s2);
        if ( v49 > 0x40 )
        {
          if ( v48 )
            j_j___libc_free_0_0(v48);
        }
      }
    }
    v19 = sub_3909460(a1);
    v20 = sub_39092A0(v19);
    v47 = 1;
    v46 = 3;
    v40 = v37;
    v41 = v20 - v37;
    s2 = "unexpected token in '.ifc' directive";
    if ( (unsigned __int8)sub_3909E20(a1, 25, &s2) )
      return 1;
    v21 = sub_38EAF10(a1);
    v47 = 1;
    v43 = v22;
    v42 = v21;
    s2 = "unexpected token in '.ifc' directive";
    v46 = 3;
    if ( (unsigned __int8)sub_3909E20(a1, 9, &s2) )
    {
      return 1;
    }
    else
    {
      v23 = 0;
      v24 = sub_16D24E0(&v42, byte_3F15413, 6, 0);
      v25 = v43;
      if ( v24 < v43 )
      {
        v23 = v43 - v24;
        v25 = v24;
      }
      n = v23;
      s2 = (void *)(v42 + v25);
      v26 = sub_16D2680((__int64 *)&s2, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      v27 = s2;
      if ( v26 > n )
        v26 = n;
      v28 = n - v23 + v26;
      if ( v28 > n )
        v28 = n;
      v29 = 0;
      v30 = sub_16D24E0(&v40, byte_3F15413, 6, 0);
      v31 = v41;
      if ( v30 < v41 )
      {
        v29 = v41 - v30;
        v31 = v30;
      }
      n = v29;
      s2 = (void *)(v40 + v31);
      v32 = sub_16D2680((__int64 *)&s2, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      v5 = 0;
      if ( v32 > n )
        v32 = n;
      v33 = n - v29 + v32;
      if ( v33 > n )
        v33 = n;
      if ( v28 == v33 )
      {
        if ( v28 )
        {
          v35 = memcmp(s2, v27, v28);
          v5 = 0;
          v36 = v35 == 0;
          v34 = a2 == (v35 == 0);
          v39 = a2 ^ v36;
        }
        else
        {
          v34 = a2;
          v39 = a2 ^ 1;
        }
      }
      else
      {
        v34 = a2 ^ 1;
      }
      *(_BYTE *)(a1 + 384) = v34;
      *(_BYTE *)(a1 + 385) = v39;
    }
  }
  else
  {
    sub_38F0630(a1);
    return 0;
  }
  return v5;
}
