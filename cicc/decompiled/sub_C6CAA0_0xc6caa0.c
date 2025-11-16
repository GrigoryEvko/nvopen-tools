// Function: sub_C6CAA0
// Address: 0xc6caa0
//
char __fastcall sub_C6CAA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 **v3; // r15
  unsigned __int64 v4; // r13
  __int64 v5; // r14
  _QWORD *v6; // rax
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 **v9; // r14
  __int64 v10; // r15
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // r9
  __int64 *v14; // rdi
  __int64 v15; // r9
  __int64 *v17; // [rsp+8h] [rbp-E8h]
  __int64 *v18; // [rsp+18h] [rbp-D8h]
  size_t v19; // [rsp+20h] [rbp-D0h]
  size_t n; // [rsp+28h] [rbp-C8h]
  __int64 v21; // [rsp+30h] [rbp-C0h]
  __int64 *v22; // [rsp+30h] [rbp-C0h]
  __int64 *v23; // [rsp+30h] [rbp-C0h]
  __int64 v24; // [rsp+30h] [rbp-C0h]
  __int64 s2; // [rsp+38h] [rbp-B8h]
  __int64 *v26; // [rsp+80h] [rbp-70h] BYREF
  __int64 v27; // [rsp+88h] [rbp-68h]
  size_t v28; // [rsp+90h] [rbp-60h]
  __int64 v29[2]; // [rsp+A0h] [rbp-50h] BYREF
  _QWORD v30[8]; // [rsp+B0h] [rbp-40h] BYREF

  v2 = -1;
  v3 = *(__int64 ***)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  v5 = *(_QWORD *)(a2 + 8);
  n = 0;
  v18 = 0;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    sub_C6B0E0(v29, -1, 0);
    sub_C6B270(&v26, (__int64)v29);
    v2 = v27;
    v18 = v26;
    n = v28;
    if ( (_QWORD *)v29[0] != v30 )
      j_j___libc_free_0(v29[0], v30[0] + 1LL);
  }
  LOBYTE(v6) = sub_C6A630((char *)0xFFFFFFFFFFFFFFFELL, 0, 0);
  if ( (_BYTE)v6 )
  {
    v19 = 0;
    v17 = 0;
    s2 = -2;
    if ( v4 )
    {
LABEL_4:
      v7 = 0;
      v8 = v5 + 24;
      v9 = v3;
      v10 = v8;
      while ( 1 )
      {
        if ( v9 )
        {
          *v9 = 0;
          v9[1] = 0;
          v9[2] = 0;
          v21 = *(_QWORD *)(v10 - 24);
          if ( v21 )
          {
            v11 = (__int64 *)sub_22077B0(32);
            if ( v11 )
            {
              v12 = v21;
              v22 = v11;
              *v11 = (__int64)(v11 + 2);
              sub_C68E20(v11, *(_BYTE **)v12, *(_QWORD *)v12 + *(_QWORD *)(v12 + 8));
              v11 = v22;
            }
            v13 = *v9;
            *v9 = v11;
            if ( v13 )
            {
              if ( (__int64 *)*v13 != v13 + 2 )
              {
                v23 = v13;
                j_j___libc_free_0(*v13, v13[2] + 1);
                v13 = v23;
              }
              j_j___libc_free_0(v13, 32);
              v11 = *v9;
            }
            v14 = (__int64 *)*v11;
            v15 = v11[1];
            v9[1] = (__int64 *)*v11;
            v9[2] = (__int64 *)v15;
          }
          else
          {
            *(__m128i *)(v9 + 1) = _mm_loadu_si128((const __m128i *)(v10 - 16));
            v15 = (__int64)v9[2];
            v14 = v9[1];
          }
        }
        else
        {
          v15 = MEMORY[0x10];
          v14 = (__int64 *)MEMORY[8];
        }
        LOBYTE(v6) = (__int64 *)((char *)v14 + 1) == 0;
        if ( v2 != -1 )
        {
          LOBYTE(v6) = (__int64 *)((char *)v14 + 2) == 0;
          if ( v2 != -2 )
          {
            LOBYTE(v6) = n;
            if ( n != v15 )
              goto LABEL_22;
            v24 = v15;
            if ( !n )
              goto LABEL_19;
            LODWORD(v6) = memcmp(v14, (const void *)v2, n);
            v15 = v24;
            LOBYTE(v6) = (_DWORD)v6 == 0;
          }
        }
        if ( (_BYTE)v6 )
          goto LABEL_19;
LABEL_22:
        LOBYTE(v6) = (__int64 *)((char *)v14 + 1) == 0;
        if ( s2 == -1 )
          goto LABEL_27;
        LOBYTE(v6) = (__int64 *)((char *)v14 + 2) == 0;
        if ( s2 == -2 )
          goto LABEL_27;
        LOBYTE(v6) = v19;
        if ( v15 != v19 )
          goto LABEL_28;
        if ( v19 )
        {
          LOBYTE(v6) = memcmp(v14, (const void *)s2, v19) == 0;
LABEL_27:
          if ( !(_BYTE)v6 )
LABEL_28:
            LOBYTE(v6) = sub_C6CEC0(v9 + 3, v10);
        }
LABEL_19:
        ++v7;
        v9 += 8;
        v10 += 64;
        if ( v4 <= v7 )
          goto LABEL_33;
      }
    }
  }
  else
  {
    sub_C6B0E0(v29, -2, 0);
    sub_C6B270(&v26, (__int64)v29);
    v17 = v26;
    s2 = v27;
    v19 = v28;
    v6 = v30;
    if ( (_QWORD *)v29[0] != v30 )
      LOBYTE(v6) = j_j___libc_free_0(v29[0], v30[0] + 1LL);
    if ( v4 )
      goto LABEL_4;
LABEL_33:
    if ( v17 )
    {
      if ( (__int64 *)*v17 != v17 + 2 )
        j_j___libc_free_0(*v17, v17[2] + 1);
      LOBYTE(v6) = j_j___libc_free_0(v17, 32);
    }
  }
  if ( v18 )
  {
    if ( (__int64 *)*v18 != v18 + 2 )
      j_j___libc_free_0(*v18, v18[2] + 1);
    LOBYTE(v6) = j_j___libc_free_0(v18, 32);
  }
  return (char)v6;
}
