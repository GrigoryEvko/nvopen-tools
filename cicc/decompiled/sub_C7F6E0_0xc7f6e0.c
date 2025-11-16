// Function: sub_C7F6E0
// Address: 0xc7f6e0
//
unsigned __int64 __fastcall sub_C7F6E0(__int64 a1, int a2, __int64 a3, char a4, double a5)
{
  __int64 v7; // r8
  char v8; // r13
  __int64 v9; // r8
  _QWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  char *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  size_t v16; // rax
  _QWORD *v17; // rcx
  size_t v18; // rdx
  unsigned __int64 result; // rax
  const char *v20; // r13
  size_t v21; // rax
  __int64 v22; // rcx
  size_t v23; // rdx
  _BYTE *v24; // rax
  __int64 v25; // rsi
  unsigned __int64 v26; // rdi
  char *v27; // rcx
  char *v28; // r13
  unsigned int v29; // ecx
  unsigned int v30; // ecx
  unsigned int v31; // eax
  __int64 v32; // [rsp+18h] [rbp-C8h]
  char *format; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v37; // [rsp+40h] [rbp-A0h]
  _BYTE v38[8]; // [rsp+48h] [rbp-98h] BYREF
  char s[32]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v40[3]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v41; // [rsp+88h] [rbp-58h]
  _WORD *v42; // [rsp+90h] [rbp-50h]
  __int64 v43; // [rsp+98h] [rbp-48h]
  char **p_format; // [rsp+A0h] [rbp-40h]

  v7 = sub_C7F6B0(a2);
  if ( a4 )
    v7 = a3;
  if ( fabs(a5) <= 1.797693134862316e308 )
  {
    v8 = 101;
    if ( a2 )
    {
      v8 = 69;
      if ( a2 != 1 )
        v8 = 102;
    }
    p_format = &format;
    v43 = 0x100000000LL;
    v32 = v7;
    v40[0] = &unk_49DD288;
    format = v38;
    v36 = 0;
    v37 = 8;
    v40[1] = 2;
    v40[2] = 0;
    v41 = 0;
    v42 = 0;
    sub_CB5980(v40, 0, 0, 0);
    v9 = v32;
    if ( (unsigned __int64)(v41 - (_QWORD)v42) <= 1 )
    {
      v11 = sub_CB6200(v40, "%.", 2);
      v9 = v32;
      v10 = (_QWORD *)v11;
    }
    else
    {
      v10 = v40;
      *v42++ = 11813;
    }
    v12 = sub_CB59D0(v10, v9);
    v13 = *(char **)(v12 + 32);
    if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
    {
      sub_CB5D20(v12, (unsigned int)v8);
    }
    else
    {
      *(_QWORD *)(v12 + 32) = v13 + 1;
      *v13 = v8;
    }
    if ( a2 == 3 )
      a5 = a5 * 100.0;
    v14 = v36;
    if ( v36 + 1 > v37 )
    {
      sub_C8D290(&format, v38, v36 + 1, 1);
      v14 = v36;
    }
    v15 = 32;
    format[v14] = 0;
    snprintf(s, 0x20u, format, a5);
    v16 = strlen(s);
    v17 = *(_QWORD **)(a1 + 32);
    v18 = v16;
    if ( v16 > *(_QWORD *)(a1 + 24) - (_QWORD)v17 )
    {
      v15 = (__int64)s;
      sub_CB6200(a1, s, v16);
      if ( a2 != 3 )
      {
LABEL_20:
        v40[0] = &unk_49DD388;
        result = sub_CB5840(v40);
        if ( format != v38 )
          return _libc_free(format, v15);
        return result;
      }
    }
    else
    {
      if ( v16 )
      {
        if ( (unsigned int)v16 >= 8 )
        {
          v26 = (unsigned __int64)(v17 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *v17 = *(_QWORD *)s;
          v15 = *(_QWORD *)&v38[(unsigned int)v16];
          *(_QWORD *)((char *)v17 + (unsigned int)v16 - 8) = v15;
          v27 = (char *)v17 - v26;
          v28 = (char *)(s - v27);
          v29 = (v16 + (_DWORD)v27) & 0xFFFFFFF8;
          if ( v29 >= 8 )
          {
            v30 = v29 & 0xFFFFFFF8;
            v31 = 0;
            do
            {
              v15 = v31;
              v31 += 8;
              *(_QWORD *)(v26 + v15) = *(_QWORD *)&v28[v15];
            }
            while ( v31 < v30 );
          }
        }
        else if ( (v16 & 4) != 0 )
        {
          *(_DWORD *)v17 = *(_DWORD *)s;
          v15 = *(unsigned int *)&v38[(unsigned int)v16 + 4];
          *(_DWORD *)((char *)v17 + (unsigned int)v16 - 4) = v15;
        }
        else if ( (_DWORD)v16 )
        {
          *(_BYTE *)v17 = s[0];
          if ( (v16 & 2) != 0 )
          {
            v15 = *(unsigned __int16 *)&v38[(unsigned int)v16 + 6];
            *(_WORD *)((char *)v17 + (unsigned int)v16 - 2) = v15;
          }
        }
        *(_QWORD *)(a1 + 32) += v18;
      }
      if ( a2 != 3 )
        goto LABEL_20;
    }
    v24 = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)v24 >= *(_QWORD *)(a1 + 24) )
    {
      v15 = 37;
      sub_CB5D20(a1, 37);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v24 + 1;
      *v24 = 37;
    }
    goto LABEL_20;
  }
  v20 = "-INF";
  if ( (_mm_movemask_pd((__m128d)*(unsigned __int64 *)&a5) & 1) == 0 )
    v20 = "INF";
  v21 = strlen(v20);
  v22 = *(_QWORD *)(a1 + 32);
  v23 = v21;
  result = *(_QWORD *)(a1 + 24) - v22;
  if ( v23 > result )
    return sub_CB6200(a1, v20, v23);
  if ( (_DWORD)v23 )
  {
    LODWORD(result) = 0;
    do
    {
      v25 = (unsigned int)result;
      result = (unsigned int)(result + 1);
      *(_BYTE *)(v22 + v25) = v20[v25];
    }
    while ( (unsigned int)result < (unsigned int)v23 );
  }
  *(_QWORD *)(a1 + 32) += v23;
  return result;
}
