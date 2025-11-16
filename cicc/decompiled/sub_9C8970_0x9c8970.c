// Function: sub_9C8970
// Address: 0x9c8970
//
__int64 __fastcall sub_9C8970(__int64 a1, __int64 a2, __int64 **a3, unsigned int a4, __int64 a5)
{
  __int64 *v9; // rdi
  unsigned __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _BYTE *v13; // r13
  size_t v14; // rdx
  void *v15; // rcx
  void *v16; // rax
  const char *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned int v23; // edx
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rdx
  unsigned int v30; // r9d
  size_t v31; // [rsp+0h] [rbp-120h]
  void *v32; // [rsp+8h] [rbp-118h]
  __int64 v33; // [rsp+18h] [rbp-108h] BYREF
  _QWORD v34[4]; // [rsp+20h] [rbp-100h] BYREF
  __int16 v35; // [rsp+40h] [rbp-E0h]
  void *s; // [rsp+50h] [rbp-D0h] BYREF
  size_t n; // [rsp+58h] [rbp-C8h]
  __int64 v38; // [rsp+60h] [rbp-C0h]
  _BYTE v39[184]; // [rsp+68h] [rbp-B8h] BYREF

  v9 = *a3;
  v10 = *((unsigned int *)a3 + 2);
  s = v39;
  n = 0;
  v38 = 128;
  if ( (unsigned __int8)sub_9C3E90((__int64)v9, v10, a4, &s)
    || (v11 = *(_QWORD *)(a2 + 744), v12 = **a3, (unsigned int)v12 >= (unsigned int)((*(_QWORD *)(a2 + 752) - v11) >> 5))
    || (v13 = *(_BYTE **)(v11 + 32LL * (unsigned int)v12 + 16)) == 0 )
  {
    HIBYTE(v35) = 1;
    v17 = "Invalid record";
    goto LABEL_8;
  }
  v14 = n;
  v15 = s;
  if ( n )
  {
    v31 = n;
    v32 = s;
    v16 = memchr(s, 0, n);
    v15 = v32;
    v14 = v31;
    if ( v16 )
    {
      HIBYTE(v35) = 1;
      v17 = "Invalid value name";
LABEL_8:
      v34[0] = v17;
      v18 = a2 + 8;
      LOBYTE(v35) = 3;
      sub_9C81F0(&v33, a2 + 8, (__int64)v34);
      v19 = v33;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v19 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_9;
    }
  }
  v18 = (__int64)v34;
  v34[1] = v14;
  v35 = 261;
  v34[0] = v15;
  sub_BD6B50(v13, v34);
  if ( (unsigned __int8)(*v13 - 2) <= 1u || !*v13 )
  {
    v21 = *(unsigned int *)(a2 + 872);
    v22 = *(_QWORD *)(a2 + 856);
    if ( (_DWORD)v21 )
    {
      v23 = (v21 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v18 = v22 + 8LL * v23;
      v24 = *(_QWORD *)v18;
      if ( v13 == *(_BYTE **)v18 )
      {
LABEL_16:
        if ( v18 != v22 + 8 * v21 )
        {
          v25 = *(unsigned int *)(a5 + 52);
          if ( (unsigned int)v25 > 8 || (v29 = 292, !_bittest64(&v29, v25)) )
          {
            v26 = *(_QWORD *)(a2 + 440);
            v27 = sub_BD5D20(v13);
            v18 = sub_BAA410(v26, v27);
            sub_B2F990(v13, v18);
          }
        }
      }
      else
      {
        v18 = 1;
        while ( v24 != -4096 )
        {
          v30 = v18 + 1;
          v23 = (v21 - 1) & (v18 + v23);
          v18 = v22 + 8LL * v23;
          v24 = *(_QWORD *)v18;
          if ( v13 == *(_BYTE **)v18 )
            goto LABEL_16;
          v18 = v30;
        }
      }
    }
  }
  v28 = *(_BYTE *)(a1 + 8);
  *(_QWORD *)a1 = v13;
  *(_BYTE *)(a1 + 8) = v28 & 0xFC | 2;
LABEL_9:
  if ( s != v39 )
    _libc_free(s, v18);
  return a1;
}
