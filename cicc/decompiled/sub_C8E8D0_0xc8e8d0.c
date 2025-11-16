// Function: sub_C8E8D0
// Address: 0xc8e8d0
//
_QWORD *__fastcall sub_C8E8D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int8 v4; // r14
  __int64 *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // r12
  size_t v11; // rbx
  __int64 v12; // rax
  char v13; // dl
  size_t v14; // rdi
  __int64 v15; // rax
  const void *v16; // r8
  size_t v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdi
  size_t v20; // rcx
  __int64 v21; // rdx
  int v23; // eax
  char *v24; // rdi
  size_t v25; // rdx
  __int64 v27; // [rsp+8h] [rbp-198h]
  const void *v28; // [rsp+18h] [rbp-188h]
  size_t v29; // [rsp+20h] [rbp-180h]
  size_t v30; // [rsp+20h] [rbp-180h]
  __int64 v33; // [rsp+48h] [rbp-158h]
  __int64 v34[4]; // [rsp+50h] [rbp-150h] BYREF
  __int16 v35; // [rsp+70h] [rbp-130h]
  char v36[32]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v37; // [rsp+A0h] [rbp-100h]
  _QWORD v38[2]; // [rsp+B0h] [rbp-F0h] BYREF
  char v39; // [rsp+C0h] [rbp-E0h]
  __int16 v40; // [rsp+D0h] [rbp-D0h]
  __int64 *v41; // [rsp+E0h] [rbp-C0h] BYREF
  size_t n; // [rsp+E8h] [rbp-B8h]
  _QWORD src[2]; // [rsp+F0h] [rbp-B0h] BYREF
  __int16 v44; // [rsp+100h] [rbp-A0h]
  __int64 *v45; // [rsp+110h] [rbp-90h] BYREF
  size_t v46; // [rsp+118h] [rbp-88h]
  unsigned __int64 v47; // [rsp+120h] [rbp-80h]
  char v48[8]; // [rsp+128h] [rbp-78h] BYREF
  __int16 v49; // [rsp+130h] [rbp-70h]

  v7 = (__int64 *)&v45;
  v49 = 260;
  v45 = (__int64 *)a3;
  sub_C7EA90((__int64)a1, (__int64 *)&v45, 0, 1u, 0, 0);
  v10 = *(__int64 **)a3;
  v11 = *(_QWORD *)(a3 + 8);
  v45 = (__int64 *)v48;
  v46 = 0;
  v47 = 64;
  if ( v11 > 0x40 )
  {
    sub_C8D290((__int64)&v45, v48, v11, 1u, v8, v9);
    v24 = (char *)v45 + v46;
  }
  else
  {
    if ( !v11 )
      goto LABEL_3;
    v24 = v48;
  }
  v7 = v10;
  memcpy(v24, v10, v11);
  v11 += v46;
LABEL_3:
  v46 = v11;
  v12 = (__int64)(*(_QWORD *)(a2 + 32) - *(_QWORD *)(a2 + 24)) >> 5;
  v13 = a1[2] & 1;
  if ( (_DWORD)v12 )
  {
    v27 = 32LL * (unsigned int)v12;
    if ( v13 )
    {
      v33 = 0;
      while ( 1 )
      {
        v14 = 0;
        v15 = *(_QWORD *)(a2 + 24) + v33;
        v16 = *(const void **)v15;
        v17 = *(_QWORD *)(v15 + 8);
        v46 = 0;
        if ( v17 > v47 )
        {
          v28 = v16;
          v30 = v17;
          sub_C8D290((__int64)&v45, v48, v17, 1u, (__int64)v16, v9);
          v14 = v46;
          v16 = v28;
          v17 = v30;
        }
        if ( v17 )
        {
          v29 = v17;
          memcpy((char *)v45 + v14, v16, v17);
          v14 = v46;
          v17 = v29;
        }
        v46 = v14 + v17;
        v44 = 257;
        v40 = 257;
        v35 = 260;
        v37 = 257;
        v34[0] = a3;
        sub_C81B70(&v45, (__int64)v34, (__int64)v36, (__int64)v38, (__int64)&v41);
        v7 = (__int64 *)&v41;
        v44 = 261;
        v41 = v45;
        n = v46;
        sub_C7EA90((__int64)v38, (__int64 *)&v41, 0, 1u, 0, v4);
        if ( (a1[2] & 1) == 0 && *a1 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*a1 + 8LL))(*a1);
        if ( (v39 & 1) == 0 )
          break;
        v23 = v38[0];
        v33 += 32;
        *((_BYTE *)a1 + 16) |= 1u;
        *(_DWORD *)a1 = v23;
        a1[1] = v38[1];
        if ( v33 == v27 )
          goto LABEL_19;
      }
      v18 = v38[0];
      *((_BYTE *)a1 + 16) &= ~1u;
      *a1 = v18;
    }
  }
  else if ( v13 )
  {
    goto LABEL_19;
  }
  v7 = v45;
  v41 = src;
  sub_C8D8C0((__int64 *)&v41, v45, (__int64)v45 + v46);
  v19 = (__int64 *)*a4;
  if ( v41 == src )
  {
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)v19 = src[0];
      }
      else
      {
        v7 = src;
        memcpy(v19, src, n);
      }
      v25 = n;
      v19 = (__int64 *)*a4;
    }
    a4[1] = v25;
    *((_BYTE *)v19 + v25) = 0;
    v19 = v41;
    goto LABEL_17;
  }
  v20 = n;
  v21 = src[0];
  if ( v19 == a4 + 2 )
  {
    v7 = a4;
    *a4 = (__int64)v41;
    a4[1] = v20;
    a4[2] = v21;
  }
  else
  {
    v7 = (__int64 *)a4[2];
    *a4 = (__int64)v41;
    a4[1] = v20;
    a4[2] = v21;
    if ( v19 )
    {
      v41 = v19;
      src[0] = v7;
      goto LABEL_17;
    }
  }
  v41 = src;
  v19 = src;
LABEL_17:
  n = 0;
  *(_BYTE *)v19 = 0;
  if ( v41 != src )
  {
    v7 = (__int64 *)(src[0] + 1LL);
    j_j___libc_free_0(v41, src[0] + 1LL);
  }
LABEL_19:
  if ( v45 != (__int64 *)v48 )
    _libc_free(v45, v7);
  return a1;
}
