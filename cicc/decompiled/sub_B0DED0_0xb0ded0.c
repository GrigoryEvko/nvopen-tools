// Function: sub_B0DED0
// Address: 0xb0ded0
//
__int64 __fastcall sub_B0DED0(_QWORD *a1, const void *a2, __int64 a3)
{
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // rax
  __int64 v5; // rax
  int v6; // r8d
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  char *v10; // r13
  int v11; // eax
  char *v12; // r9
  char *v13; // r10
  size_t v14; // r11
  __int64 v15; // r13
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 *v19; // rsi
  __int64 v20; // rdx
  __int64 *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r12
  signed __int64 v25; // r12
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rbx
  unsigned __int64 v29; // r8
  size_t v31; // [rsp+8h] [rbp-108h]
  char *v32; // [rsp+10h] [rbp-100h]
  char *v33; // [rsp+18h] [rbp-F8h]
  unsigned __int64 *v35; // [rsp+30h] [rbp-E0h]
  __int64 v36; // [rsp+38h] [rbp-D8h]
  void *src; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 *v38; // [rsp+48h] [rbp-C8h] BYREF
  __int64 *v39; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+58h] [rbp-B8h]
  _BYTE v41[176]; // [rsp+60h] [rbp-B0h] BYREF

  v3 = (unsigned __int64 *)a1[3];
  v39 = (__int64 *)v41;
  v40 = 0x1000000000LL;
  v4 = (unsigned __int64 *)a1[2];
  v36 = a3;
  v35 = v3;
  v38 = v4;
  if ( v3 == v4 )
  {
    v25 = 8 * a3;
    v26 = 16;
    v27 = 0;
    v28 = v25 >> 3;
    v29 = v25 >> 3;
  }
  else
  {
    do
    {
      v25 = 8 * v36;
      src = v38;
      v8 = *v4;
      v9 = (8 * v36) >> 3;
      v28 = v9;
      if ( v8 == 4096 || v8 == 159 )
      {
        v17 = (unsigned int)v40;
        v18 = (unsigned int)v40 + v9;
        if ( v18 > HIDWORD(v40) )
        {
          sub_C8D5F0(&v39, v41, v18, 8);
          v17 = (unsigned int)v40;
        }
        if ( v25 )
        {
          memcpy(&v39[v17], a2, v25);
          LODWORD(v17) = v40;
        }
        v28 = 0;
        v25 = 0;
        v36 = 0;
        LODWORD(v40) = v9 + v17;
        a2 = 0;
      }
      v10 = (char *)src;
      v11 = sub_AF4160((unsigned __int64 **)&src);
      v12 = (char *)src;
      v13 = &v10[8 * v11];
      v5 = (unsigned int)v40;
      v14 = v13 - (_BYTE *)src;
      v15 = (v13 - (_BYTE *)src) >> 3;
      v16 = v15 + (unsigned int)v40;
      if ( v16 > HIDWORD(v40) )
      {
        v31 = v13 - (_BYTE *)src;
        v32 = (char *)src;
        v33 = v13;
        sub_C8D5F0(&v39, v41, v16, 8);
        v5 = (unsigned int)v40;
        v14 = v31;
        v12 = v32;
        v13 = v33;
      }
      if ( v13 != v12 )
      {
        memcpy(&v39[v5], v12, v14);
        LODWORD(v5) = v40;
      }
      v6 = v15 + v5;
      v7 = v38;
      LODWORD(v40) = v6;
      v4 = &v7[(unsigned int)sub_AF4160(&v38)];
      v38 = v4;
    }
    while ( v35 != v4 );
    v27 = (unsigned int)v40;
    v26 = HIDWORD(v40);
    v29 = v28 + (unsigned int)v40;
  }
  if ( v29 > v26 )
  {
    sub_C8D5F0(&v39, v41, v29, 8);
    v27 = (unsigned int)v40;
  }
  v19 = v39;
  if ( v25 )
  {
    memcpy(&v39[v27], a2, v25);
    v19 = v39;
    LODWORD(v27) = v40;
  }
  LODWORD(v40) = v27 + v28;
  v20 = (unsigned int)(v27 + v28);
  v21 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[1] & 4) != 0 )
    v21 = (__int64 *)*v21;
  v22 = sub_B0D000(v21, v19, v20, 0, 1);
  v23 = sub_E3D320(v22);
  if ( v39 != (__int64 *)v41 )
    _libc_free(v39, v19);
  return v23;
}
