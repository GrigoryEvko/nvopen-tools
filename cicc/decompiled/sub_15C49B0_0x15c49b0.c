// Function: sub_15C49B0
// Address: 0x15c49b0
//
__int64 __fastcall sub_15C49B0(_QWORD *a1, const void *a2, __int64 a3)
{
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r13
  char *v10; // r13
  int v11; // eax
  char *v12; // r9
  char *v13; // rax
  size_t v14; // r10
  unsigned __int64 v15; // r13
  __int64 v16; // rdx
  void *v17; // rsi
  __int64 v18; // rdx
  __int64 *v19; // rdi
  __int64 v20; // r12
  signed __int64 v22; // r12
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rbx
  size_t v27; // [rsp+8h] [rbp-108h]
  char *v28; // [rsp+10h] [rbp-100h]
  char *v29; // [rsp+18h] [rbp-F8h]
  unsigned __int64 *v31; // [rsp+30h] [rbp-E0h]
  __int64 v32; // [rsp+38h] [rbp-D8h]
  void *src; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 *v34; // [rsp+48h] [rbp-C8h] BYREF
  _BYTE *v35; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-B8h]
  _BYTE v37[176]; // [rsp+60h] [rbp-B0h] BYREF

  v3 = (unsigned __int64 *)a1[4];
  v35 = v37;
  v36 = 0x1000000000LL;
  v4 = (unsigned __int64 *)a1[3];
  v32 = a3;
  v31 = v3;
  v34 = v4;
  if ( v3 == v4 )
  {
    v22 = 8 * a3;
    v23 = 16;
    v24 = 0;
    v25 = v22 >> 3;
  }
  else
  {
    do
    {
      v22 = 8 * v32;
      src = v34;
      v8 = *v4;
      v9 = (8 * v32) >> 3;
      v25 = v9;
      if ( v8 == 4096 || v8 == 159 )
      {
        v16 = (unsigned int)v36;
        if ( HIDWORD(v36) - (unsigned __int64)(unsigned int)v36 < v9 )
        {
          sub_16CD150(&v35, v37, v9 + (unsigned int)v36, 8);
          v16 = (unsigned int)v36;
        }
        if ( v22 )
        {
          memcpy(&v35[8 * v16], a2, v22);
          LODWORD(v16) = v36;
        }
        v25 = 0;
        v22 = 0;
        v32 = 0;
        LODWORD(v36) = v9 + v16;
        a2 = 0;
      }
      v10 = (char *)src;
      v11 = sub_15B11B0((unsigned __int64 **)&src);
      v12 = (char *)src;
      v5 = (unsigned int)v36;
      v13 = &v10[8 * v11];
      v14 = v13 - (_BYTE *)src;
      v15 = (v13 - (_BYTE *)src) >> 3;
      if ( v15 > HIDWORD(v36) - (unsigned __int64)(unsigned int)v36 )
      {
        v27 = v13 - (_BYTE *)src;
        v28 = (char *)src;
        v29 = v13;
        sub_16CD150(&v35, v37, v15 + (unsigned int)v36, 8);
        v5 = (unsigned int)v36;
        v14 = v27;
        v12 = v28;
        v13 = v29;
      }
      if ( v13 != v12 )
      {
        memcpy(&v35[8 * v5], v12, v14);
        LODWORD(v5) = v36;
      }
      v6 = v15 + v5;
      v7 = v34;
      LODWORD(v36) = v6;
      v4 = &v7[(unsigned int)sub_15B11B0(&v34)];
      v34 = v4;
    }
    while ( v31 != v4 );
    v24 = (unsigned int)v36;
    v23 = HIDWORD(v36) - (unsigned __int64)(unsigned int)v36;
  }
  if ( v25 > v23 )
  {
    sub_16CD150(&v35, v37, v25 + v24, 8);
    v24 = (unsigned int)v36;
  }
  v17 = v35;
  if ( v22 )
  {
    memcpy(&v35[8 * v24], a2, v22);
    v17 = v35;
    LODWORD(v24) = v36;
  }
  LODWORD(v36) = v24 + v25;
  v18 = (unsigned int)(v24 + v25);
  v19 = (__int64 *)(a1[2] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[2] & 4) != 0 )
    v19 = (__int64 *)*v19;
  v20 = sub_15C4420(v19, v17, v18, 0, 1);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v20;
}
