// Function: sub_C6D3C0
// Address: 0xc6d3c0
//
char __fastcall sub_C6D3C0(__int64 a1, size_t a2, __int64 a3)
{
  size_t v3; // r12
  __int64 **v4; // r15
  __int64 **v5; // rbx
  __int64 *v6; // r14
  __int64 *v7; // rax
  __int64 *v8; // r12
  __int64 *v9; // r8
  __int64 v10; // rax
  size_t v11; // r15
  __int64 v12; // rbx
  _QWORD *v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // r8
  _QWORD *v18; // r8
  char *v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rdx
  _QWORD *v22; // r8
  __int64 *v24; // [rsp+10h] [rbp-E0h]
  __int64 *v25; // [rsp+18h] [rbp-D8h]
  __int64 *v26; // [rsp+18h] [rbp-D8h]
  __int64 v28; // [rsp+20h] [rbp-D0h]
  size_t n; // [rsp+28h] [rbp-C8h]
  size_t na; // [rsp+28h] [rbp-C8h]
  size_t v31; // [rsp+30h] [rbp-C0h]
  __int64 v33; // [rsp+38h] [rbp-B8h]
  __int64 v34; // [rsp+38h] [rbp-B8h]
  _QWORD *v35; // [rsp+38h] [rbp-B8h]
  _QWORD *v36; // [rsp+38h] [rbp-B8h]
  __int64 v37; // [rsp+38h] [rbp-B8h]
  __int64 *v38; // [rsp+80h] [rbp-70h] BYREF
  __int64 v39; // [rsp+88h] [rbp-68h]
  size_t v40; // [rsp+90h] [rbp-60h]
  __int64 v41[2]; // [rsp+A0h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+B0h] [rbp-40h] BYREF

  v3 = a2;
  *(_QWORD *)(a1 + 16) = 0;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    sub_C6B0E0(v41, -1, 0);
    sub_C6B270(&v38, (__int64)v41);
    v6 = v38;
    v33 = v39;
    v31 = v40;
    if ( (_QWORD *)v41[0] != v42 )
      j_j___libc_free_0(v41[0], v42[0] + 1LL);
    v4 = *(__int64 ***)(a1 + 8);
    v5 = &v4[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
    if ( v4 == v5 )
    {
LABEL_16:
      if ( v6 )
      {
        if ( (__int64 *)*v6 != v6 + 2 )
          j_j___libc_free_0(*v6, v6[2] + 1);
        j_j___libc_free_0(v6, 32);
      }
      goto LABEL_20;
    }
    n = a2;
    while ( 1 )
    {
LABEL_12:
      while ( !v4 )
      {
LABEL_11:
        v4 += 8;
        if ( v4 == v5 )
          goto LABEL_15;
      }
      *v4 = 0;
      v4[1] = 0;
      v4[2] = 0;
      if ( v6 )
      {
        v7 = (__int64 *)sub_22077B0(32);
        v8 = v7;
        if ( v7 )
        {
          *v7 = (__int64)(v7 + 2);
          sub_C68E20(v7, (_BYTE *)*v6, *v6 + v6[1]);
        }
        v9 = *v4;
        *v4 = v8;
        if ( v9 )
        {
          if ( (__int64 *)*v9 != v9 + 2 )
          {
            v25 = v9;
            j_j___libc_free_0(*v9, v9[2] + 1);
            v9 = v25;
          }
          j_j___libc_free_0(v9, 32);
          v8 = *v4;
        }
        v10 = v8[1];
        v4[1] = (__int64 *)*v8;
        v4[2] = (__int64 *)v10;
        goto LABEL_11;
      }
      v4 += 8;
      *(v4 - 7) = (__int64 *)v33;
      *(v4 - 6) = (__int64 *)v31;
      if ( v4 == v5 )
      {
LABEL_15:
        v3 = n;
        goto LABEL_16;
      }
    }
  }
  v4 = *(__int64 ***)(a1 + 8);
  v5 = &v4[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
  if ( v5 != v4 )
  {
    v33 = -1;
    v6 = 0;
    v31 = 0;
    n = a2;
    goto LABEL_12;
  }
LABEL_20:
  v11 = 0;
  v12 = -1;
  v26 = 0;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    sub_C6B0E0(v41, -1, 0);
    sub_C6B270(&v38, (__int64)v41);
    v12 = v39;
    v11 = v40;
    v26 = v38;
    if ( (_QWORD *)v41[0] != v42 )
      j_j___libc_free_0(v41[0], v42[0] + 1LL);
  }
  LOBYTE(v13) = sub_C6A630((char *)0xFFFFFFFFFFFFFFFELL, 0, 0);
  if ( (_BYTE)v13 )
  {
    v24 = 0;
    v14 = -2;
    na = 0;
    if ( v3 != a3 )
    {
LABEL_23:
      v15 = v3;
      v16 = a3;
      while ( 1 )
      {
        v19 = *(char **)(v15 + 8);
        v17 = *(_QWORD *)(v15 + 16);
        LOBYTE(v13) = v19 + 1 == 0;
        if ( v12 != -1 )
        {
          LOBYTE(v13) = v19 + 2 == 0;
          if ( v12 != -2 )
          {
            if ( v17 != v11 )
              goto LABEL_35;
            v34 = *(_QWORD *)(v15 + 16);
            if ( !v11 )
              goto LABEL_27;
            LODWORD(v13) = memcmp(v19, (const void *)v12, v11);
            v17 = v34;
            LOBYTE(v13) = (_DWORD)v13 == 0;
          }
        }
        if ( (_BYTE)v13 )
          goto LABEL_27;
LABEL_35:
        LOBYTE(v13) = v19 + 1 == 0;
        if ( v14 == -1 )
          goto LABEL_40;
        LOBYTE(v13) = v19 + 2 == 0;
        if ( v14 == -2 )
          goto LABEL_40;
        LOBYTE(v13) = na;
        if ( v17 != na )
          goto LABEL_41;
        if ( na )
        {
          LOBYTE(v13) = memcmp(v19, (const void *)v14, na) == 0;
LABEL_40:
          if ( !(_BYTE)v13 )
          {
LABEL_41:
            sub_C6BF30(a1, v15, v41);
            v20 = v41[0];
            v21 = *(_QWORD **)v15;
            *(_QWORD *)v15 = 0;
            v22 = *(_QWORD **)v20;
            *(_QWORD *)v20 = v21;
            if ( v22 )
            {
              if ( (_QWORD *)*v22 != v22 + 2 )
              {
                v28 = v20;
                v36 = v22;
                j_j___libc_free_0(*v22, v22[2] + 1LL);
                v20 = v28;
                v22 = v36;
              }
              v37 = v20;
              j_j___libc_free_0(v22, 32);
              v20 = v37;
            }
            *(__m128i *)(v20 + 8) = _mm_loadu_si128((const __m128i *)(v15 + 8));
            sub_C6A4F0(v41[0] + 24, (unsigned __int16 *)(v15 + 24));
            ++*(_DWORD *)(a1 + 16);
            LOBYTE(v13) = sub_C6BC50((unsigned __int16 *)(v15 + 24));
          }
        }
LABEL_27:
        v18 = *(_QWORD **)v15;
        if ( *(_QWORD *)v15 )
        {
          if ( (_QWORD *)*v18 != v18 + 2 )
          {
            v35 = *(_QWORD **)v15;
            j_j___libc_free_0(*v18, v18[2] + 1LL);
            v18 = v35;
          }
          LOBYTE(v13) = j_j___libc_free_0(v18, 32);
        }
        v15 += 64;
        if ( v16 == v15 )
          goto LABEL_49;
      }
    }
  }
  else
  {
    sub_C6B0E0(v41, -2, 0);
    sub_C6B270(&v38, (__int64)v41);
    v14 = v39;
    v24 = v38;
    na = v40;
    v13 = v42;
    if ( (_QWORD *)v41[0] != v42 )
      LOBYTE(v13) = j_j___libc_free_0(v41[0], v42[0] + 1LL);
    if ( v3 != a3 )
      goto LABEL_23;
LABEL_49:
    if ( v24 )
    {
      if ( (__int64 *)*v24 != v24 + 2 )
        j_j___libc_free_0(*v24, v24[2] + 1);
      LOBYTE(v13) = j_j___libc_free_0(v24, 32);
    }
  }
  if ( v26 )
  {
    if ( (__int64 *)*v26 != v26 + 2 )
      j_j___libc_free_0(*v26, v26[2] + 1);
    LOBYTE(v13) = j_j___libc_free_0(v26, 32);
  }
  return (char)v13;
}
