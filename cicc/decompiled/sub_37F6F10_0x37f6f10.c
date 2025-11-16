// Function: sub_37F6F10
// Address: 0x37f6f10
//
void __fastcall sub_37F6F10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned __int64 v9; // r9
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *i; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *j; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  _BYTE *v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  int v26; // r13d
  _BYTE *v27; // rsi
  _BYTE *v28; // rdi
  __int64 v29; // r15
  __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  unsigned __int64 *v32; // r12
  unsigned __int64 v33; // rdi
  __int64 *v34; // rbx
  __int64 *v35; // r13
  __int64 v36; // rax
  unsigned __int64 *v37; // rax
  unsigned __int64 v38; // r15
  __int64 v39; // rcx
  __int64 v40; // r12
  __int64 v41; // r13
  unsigned __int64 v42; // rdi
  unsigned __int64 *v43; // [rsp+0h] [rbp-E0h]
  __int64 v44; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v45[2]; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE v46[64]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE *v47; // [rsp+60h] [rbp-80h] BYREF
  __int64 v48; // [rsp+68h] [rbp-78h]
  _BYTE src[112]; // [rsp+70h] [rbp-70h] BYREF

  v6 = *(_QWORD **)(a1 + 200);
  v7 = *(unsigned int *)(a1 + 504);
  *(_DWORD *)(a1 + 304) = *(_DWORD *)(*(_QWORD *)(a1 + 208) + 44LL);
  v8 = 0xCCCCCCCCCCCCCCCDLL;
  *(_DWORD *)(a1 + 308) = -858993459 * ((__int64)(*(_QWORD *)(v6[6] + 16LL) - *(_QWORD *)(v6[6] + 8LL)) >> 3);
  *(_DWORD *)(a1 + 312) = -*(_DWORD *)(v6[6] + 32LL);
  v9 = (unsigned int)((__int64)(v6[13] - v6[12]) >> 3);
  v44 = (__int64)(v6[13] - v6[12]) >> 3;
  LODWORD(v10) = v44;
  if ( v9 != v7 )
  {
    v11 = 24 * v9;
    if ( v9 < v7 )
    {
      v8 = *(_QWORD *)(a1 + 496);
      v32 = (unsigned __int64 *)(v8 + 24 * v7);
      v43 = (unsigned __int64 *)(v8 + v11);
      if ( v32 != (unsigned __int64 *)(v8 + v11) )
      {
        do
        {
          v33 = *(v32 - 3);
          v34 = (__int64 *)*(v32 - 2);
          v32 -= 3;
          v35 = (__int64 *)v33;
          if ( v34 != (__int64 *)v33 )
          {
            do
            {
              v36 = *v35;
              if ( *v35 )
              {
                if ( (v36 & 1) != 0 )
                {
                  v37 = (unsigned __int64 *)(v36 & 0xFFFFFFFFFFFFFFFELL);
                  v38 = (unsigned __int64)v37;
                  if ( v37 )
                  {
                    if ( (unsigned __int64 *)*v37 != v37 + 2 )
                      _libc_free(*v37);
                    j_j___libc_free_0(v38);
                  }
                }
              }
              ++v35;
            }
            while ( v34 != v35 );
            v33 = *v32;
          }
          if ( v33 )
            j_j___libc_free_0(v33);
        }
        while ( v43 != v32 );
        v6 = *(_QWORD **)(a1 + 200);
      }
      *(_DWORD *)(a1 + 504) = v44;
      v10 = (__int64)(v6[13] - v6[12]) >> 3;
      v7 = (unsigned int)v10;
    }
    else
    {
      if ( v9 > *(unsigned int *)(a1 + 508) )
      {
        sub_37F6CB0(a1 + 496, v9, (__int64)v6, 0xCCCCCCCCCCCCCCCDLL, a5, v9);
        v7 = *(unsigned int *)(a1 + 504);
      }
      v12 = *(_QWORD *)(a1 + 496);
      v13 = (_QWORD *)(v12 + 24 * v7);
      for ( i = (_QWORD *)(v11 + v12); i != v13; v13 += 3 )
      {
        if ( v13 )
        {
          *v13 = 0;
          v13[1] = 0;
          v13[2] = 0;
        }
      }
      v6 = *(_QWORD **)(a1 + 200);
      *(_DWORD *)(a1 + 504) = v44;
      v10 = (__int64)(v6[13] - v6[12]) >> 3;
      v7 = (unsigned int)v10;
    }
  }
  v15 = *(unsigned int *)(a1 + 352);
  if ( v15 != v7 )
  {
    v16 = 24 * v7;
    if ( v15 > v7 )
    {
      v39 = *(_QWORD *)(a1 + 344);
      v40 = v39 + 24 * v15;
      v41 = v39 + v16;
      if ( v40 != v41 )
      {
        do
        {
          v42 = *(_QWORD *)(v40 - 24);
          v40 -= 24;
          if ( v42 )
            j_j___libc_free_0(v42);
        }
        while ( v41 != v40 );
        v6 = *(_QWORD **)(a1 + 200);
      }
      *(_DWORD *)(a1 + 352) = v10;
    }
    else
    {
      v17 = *(unsigned int *)(a1 + 356);
      if ( v17 < v7 )
      {
        sub_37F6E10(a1 + 344, v7, v17, v8, a5, v9);
        v15 = *(unsigned int *)(a1 + 352);
      }
      v18 = *(_QWORD *)(a1 + 344);
      v19 = (_QWORD *)(v18 + 24 * v15);
      for ( j = (_QWORD *)(v16 + v18); j != v19; v19 += 3 )
      {
        if ( v19 )
        {
          *v19 = 0;
          v19[1] = 0;
          v19[2] = 0;
        }
      }
      *(_DWORD *)(a1 + 352) = v10;
      v6 = *(_QWORD **)(a1 + 200);
    }
  }
  v45[1] = 0x400000000LL;
  v45[0] = (unsigned __int64)v46;
  sub_384C4A0(&v47, v45, v6);
  v23 = v47;
  if ( v47 == src )
  {
    v24 = (unsigned int)v48;
    v25 = *(unsigned int *)(a1 + 232);
    v26 = v48;
    if ( (unsigned int)v48 <= v25 )
    {
      v28 = src;
      if ( (_DWORD)v48 )
      {
        memmove(*(void **)(a1 + 224), src, 16LL * (unsigned int)v48);
        v28 = v47;
      }
    }
    else
    {
      if ( (unsigned int)v48 > (unsigned __int64)*(unsigned int *)(a1 + 236) )
      {
        *(_DWORD *)(a1 + 232) = 0;
        sub_C8D5F0(a1 + 224, (const void *)(a1 + 240), v24, 0x10u, v21, v22);
        v28 = v47;
        v24 = (unsigned int)v48;
        v25 = 0;
        v27 = v47;
      }
      else
      {
        v27 = src;
        v28 = src;
        v29 = 16 * v25;
        if ( *(_DWORD *)(a1 + 232) )
        {
          memmove(*(void **)(a1 + 224), src, 16 * v25);
          v28 = v47;
          v24 = (unsigned int)v48;
          v25 = v29;
          v27 = &v47[v29];
        }
      }
      v30 = 16 * v24;
      if ( v27 != &v28[v30] )
      {
        memcpy((void *)(v25 + *(_QWORD *)(a1 + 224)), v27, v30 - v25);
        v28 = v47;
      }
    }
    *(_DWORD *)(a1 + 232) = v26;
    if ( v28 != src )
      _libc_free((unsigned __int64)v28);
  }
  else
  {
    v31 = *(_QWORD *)(a1 + 224);
    if ( v31 != a1 + 240 )
    {
      _libc_free(v31);
      v23 = v47;
    }
    *(_QWORD *)(a1 + 224) = v23;
    *(_QWORD *)(a1 + 232) = v48;
  }
  if ( (_BYTE *)v45[0] != v46 )
    _libc_free(v45[0]);
}
