// Function: sub_D10300
// Address: 0xd10300
//
__int64 __fastcall sub_D10300(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // r13
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  char *v15; // r12
  __int64 v16; // r14
  char *i; // r15
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // rbx
  const char *v21; // rax
  void *v22; // rdx
  void *v23; // r8
  bool v24; // cc
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // rdi
  char *v31; // r13
  __int64 v32; // [rsp+8h] [rbp-D8h]
  const char *s2; // [rsp+10h] [rbp-D0h]
  char *src; // [rsp+18h] [rbp-C8h]
  void *srcc; // [rsp+18h] [rbp-C8h]
  void *srca; // [rsp+18h] [rbp-C8h]
  void *srcb; // [rsp+18h] [rbp-C8h]
  char *v38; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+28h] [rbp-B8h]
  _BYTE v40[176]; // [rsp+30h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 48);
  v38 = v40;
  v32 = (__int64)a2;
  v39 = 0x1000000000LL;
  if ( v6 > 0x10 )
  {
    a2 = v40;
    v8 = a1 + 16;
    sub_C8D5F0((__int64)&v38, v40, v6, 8u, a5, a6);
    result = (unsigned int)v39;
    v7 = *(_QWORD *)(a1 + 32);
    v11 = (unsigned int)v39;
    if ( v7 == a1 + 16 )
      goto LABEL_8;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 32);
    v8 = a1 + 16;
    result = 0;
    if ( a1 + 16 == v7 )
    {
      v31 = v40;
      goto LABEL_29;
    }
  }
  while ( 1 )
  {
    v10 = *(_QWORD *)(v7 + 40);
    if ( result + 1 > (unsigned __int64)HIDWORD(v39) )
    {
      a2 = v40;
      srcb = (void *)v7;
      sub_C8D5F0((__int64)&v38, v40, result + 1, 8u, v7, a6);
      result = (unsigned int)v39;
      v7 = (__int64)srcb;
    }
    *(_QWORD *)&v38[8 * result] = v10;
    v11 = (unsigned int)(v39 + 1);
    LODWORD(v39) = v39 + 1;
    result = sub_220EF30(v7);
    v7 = result;
    if ( result == v8 )
      break;
    result = (unsigned int)v11;
  }
LABEL_8:
  v13 = 8 * v11;
  v31 = &v38[8 * v11];
  if ( v38 != v31 )
  {
    src = v38;
    _BitScanReverse64(&v14, v13 >> 3);
    sub_D0F410(v38, (__int64 *)v31, 2LL * (int)(63 - (v14 ^ 0x3F)), v12, v7);
    if ( (unsigned __int64)v13 <= 0x80 )
    {
      a2 = v31;
      sub_D0F240(src, v31);
      goto LABEL_26;
    }
    v15 = src + 128;
    a2 = src + 128;
    sub_D0F240(src, src + 128);
    if ( v31 != src + 128 )
    {
LABEL_11:
      v16 = *(_QWORD *)v15;
      for ( i = v15; ; i -= 8 )
      {
        v27 = *((_QWORD *)i - 1);
        srca = *(void **)(v16 + 8);
        v28 = *(_QWORD *)(v27 + 8);
        if ( srca )
        {
          if ( !v28 )
            goto LABEL_23;
          v18 = sub_BD5D20(v28);
          v20 = v19;
          s2 = v18;
          v21 = sub_BD5D20((__int64)srca);
          v23 = v22;
          v24 = (unsigned __int64)v22 <= v20;
          v25 = v20;
          if ( v24 )
            v25 = (size_t)v23;
          if ( v25 && (a2 = s2, srcc = v23, v26 = memcmp(v21, s2, v25), v23 = srcc, v26) )
          {
            if ( v26 >= 0 )
            {
              v15 += 8;
              *(_QWORD *)i = v16;
              if ( v31 != v15 )
                goto LABEL_11;
              break;
            }
          }
          else if ( v23 == (void *)v20 || (unsigned __int64)v23 >= v20 )
          {
LABEL_23:
            v15 += 8;
            *(_QWORD *)i = v16;
            if ( v31 != v15 )
              goto LABEL_11;
            break;
          }
          v27 = *((_QWORD *)i - 1);
        }
        else if ( !v28 )
        {
          goto LABEL_23;
        }
        *(_QWORD *)i = v27;
      }
    }
LABEL_26:
    v29 = (unsigned __int64 *)v38;
    result = (unsigned int)v39;
    v31 = &v38[8 * (unsigned int)v39];
    if ( v38 != v31 )
    {
      do
      {
        v30 = *v29;
        a2 = (_BYTE *)v32;
        ++v29;
        result = sub_D0FDF0(v30, v32);
      }
      while ( v31 != (char *)v29 );
      v31 = v38;
    }
  }
LABEL_29:
  if ( v31 != v40 )
    return _libc_free(v31, a2);
  return result;
}
