// Function: sub_16F39A0
// Address: 0x16f39a0
//
unsigned __int64 __fastcall sub_16F39A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  unsigned __int64 result; // rax
  __int64 v7; // r13
  __int64 *v8; // rax
  __int64 *v9; // r15
  __int64 v10; // rax
  char v11; // r15
  __int64 v12; // rbx
  __int64 **v13; // r14
  __int64 v14; // rax
  __int64 *v15; // rax
  unsigned __int8 **v16; // r14
  void *v17; // r9
  size_t v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  bool v21; // al
  unsigned __int8 **v22; // r9
  void *v23; // r10
  size_t v24; // rdx
  __int64 v25; // rax
  char *v26; // rdi
  int v27; // eax
  unsigned __int8 **v28; // [rsp+8h] [rbp-D8h]
  size_t n; // [rsp+10h] [rbp-D0h]
  size_t na; // [rsp+10h] [rbp-D0h]
  __int64 *v31; // [rsp+20h] [rbp-C0h]
  __int64 v32; // [rsp+20h] [rbp-C0h]
  void *v33; // [rsp+20h] [rbp-C0h]
  void *v34; // [rsp+20h] [rbp-C0h]
  unsigned __int8 **v35; // [rsp+20h] [rbp-C0h]
  unsigned __int8 **v36; // [rsp+20h] [rbp-C0h]
  unsigned __int8 **v37; // [rsp+70h] [rbp-70h] BYREF
  void *s2; // [rsp+78h] [rbp-68h]
  size_t v39; // [rsp+80h] [rbp-60h]
  __int64 v40[2]; // [rsp+90h] [rbp-50h] BYREF
  _QWORD v41[8]; // [rsp+A0h] [rbp-40h] BYREF

  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v7 = 0;
    while ( 1 )
    {
      v12 = v7 << 6;
      v13 = (__int64 **)(*(_QWORD *)(a1 + 8) + (v7 << 6));
      if ( v13 )
      {
        a4 = a2;
        v14 = *(_QWORD *)(a2 + 8);
        *v13 = 0;
        v13[1] = 0;
        v15 = (__int64 *)(v12 + v14);
        v13[2] = 0;
        v32 = *v15;
        if ( *v15 )
        {
          v8 = (__int64 *)sub_22077B0(32);
          v9 = v8;
          if ( v8 )
          {
            *v8 = (__int64)(v8 + 2);
            sub_16F1520(v8, *(_BYTE **)v32, *(_QWORD *)v32 + *(_QWORD *)(v32 + 8));
          }
          a5 = *v13;
          *v13 = v9;
          if ( a5 )
          {
            if ( (__int64 *)*a5 != a5 + 2 )
            {
              v31 = a5;
              j_j___libc_free_0(*a5, a5[2] + 1);
              a5 = v31;
            }
            j_j___libc_free_0(a5, 32);
            v9 = *v13;
          }
          v10 = v9[1];
          v13[1] = (__int64 *)*v9;
          v13[2] = (__int64 *)v10;
        }
        else
        {
          *(__m128i *)(v13 + 1) = _mm_loadu_si128((const __m128i *)(v15 + 1));
        }
      }
      v11 = sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, a4, (int)a5);
      if ( !v11 )
        break;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 8) + v12 + 8) != -1 )
      {
        v16 = 0;
LABEL_22:
        v11 = sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0, 0, a4, (int)a5);
        if ( v11 )
        {
          v11 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + v12 + 8) != -2;
          goto LABEL_24;
        }
        sub_16F2420(v40, (unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0);
        sub_16F25B0(&v37, (__int64)v40);
        v22 = v37;
        v23 = s2;
        v24 = v39;
        if ( (_QWORD *)v40[0] != v41 )
        {
          v28 = v37;
          na = v39;
          v34 = s2;
          j_j___libc_free_0(v40[0], v41[0] + 1LL);
          v22 = v28;
          v24 = na;
          v23 = v34;
        }
        v25 = v12 + *(_QWORD *)(a1 + 8);
        v26 = *(char **)(v25 + 8);
        LOBYTE(a4) = v26 + 1 == 0;
        if ( v23 != (void *)-1LL )
        {
          if ( v23 != (void *)-2LL )
          {
            if ( *(_QWORD *)(v25 + 16) == v24 )
            {
              if ( v24 )
              {
                v36 = v22;
                v27 = memcmp(v26, v23, v24);
                v22 = v36;
                v11 = v27 != 0;
              }
            }
            else
            {
              v11 = 1;
            }
            goto LABEL_43;
          }
          LOBYTE(a4) = v26 + 2 == 0;
        }
        a4 = (unsigned int)a4 ^ 1;
        v11 = a4;
LABEL_43:
        if ( v22 )
        {
          if ( *v22 != (unsigned __int8 *)(v22 + 2) )
          {
            v35 = v22;
            j_j___libc_free_0(*v22, v22[2] + 1);
            v22 = v35;
          }
          j_j___libc_free_0(v22, 32);
        }
LABEL_24:
        if ( v16 )
        {
LABEL_25:
          if ( *v16 != (unsigned __int8 *)(v16 + 2) )
            j_j___libc_free_0(*v16, v16[2] + 1);
          j_j___libc_free_0(v16, 32);
        }
        if ( v11 )
          sub_16F3DE0(*(_QWORD *)(a1 + 8) + v12 + 24, v12 + *(_QWORD *)(a2 + 8) + 24);
      }
LABEL_12:
      result = *(unsigned int *)(a1 + 24);
      if ( result <= ++v7 )
        return result;
    }
    sub_16F2420(v40, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
    sub_16F25B0(&v37, (__int64)v40);
    v16 = v37;
    v17 = s2;
    v18 = v39;
    if ( (_QWORD *)v40[0] != v41 )
    {
      n = v39;
      v33 = s2;
      j_j___libc_free_0(v40[0], v41[0] + 1LL);
      v18 = n;
      v17 = v33;
    }
    v19 = v12 + *(_QWORD *)(a1 + 8);
    v20 = *(char **)(v19 + 8);
    if ( v17 == (void *)-1LL )
    {
      v21 = v20 + 1 == 0;
    }
    else
    {
      if ( v17 != (void *)-2LL )
      {
        if ( *(_QWORD *)(v19 + 16) != v18 || v18 && memcmp(v20, v17, v18) )
          goto LABEL_22;
        goto LABEL_32;
      }
      v21 = v20 + 2 == 0;
    }
    if ( !v21 )
      goto LABEL_22;
LABEL_32:
    if ( v16 )
      goto LABEL_25;
    goto LABEL_12;
  }
  return result;
}
