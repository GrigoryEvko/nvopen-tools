// Function: sub_D39570
// Address: 0xd39570
//
void __fastcall sub_D39570(__int64 a1, unsigned int *a2)
{
  __int64 v3; // rdi
  unsigned __int64 v5; // rsi
  size_t v6; // rdx
  size_t v7; // r14
  unsigned int *v8; // rax
  __int64 v9; // r12
  char **v10; // rbx
  unsigned int *v11; // r13
  __int64 v12; // r9
  char v13; // cl
  int v14; // eax
  char **v15; // r12
  char **v16; // rbx
  char **v17; // rdi
  __int64 v18; // rcx
  _BYTE *v19; // r10
  unsigned __int64 v20; // rdx
  unsigned int v21; // eax
  _QWORD *v22; // r8
  __int64 v23; // rax
  void *v24; // rdi
  unsigned int v25; // r11d
  size_t v26; // rdx
  unsigned int v27; // eax
  unsigned int v28; // [rsp+0h] [rbp-270h]
  unsigned int v29; // [rsp+0h] [rbp-270h]
  unsigned int v30; // [rsp+0h] [rbp-270h]
  _QWORD *v31; // [rsp+8h] [rbp-268h]
  _QWORD *v32; // [rsp+8h] [rbp-268h]
  unsigned int v33; // [rsp+8h] [rbp-268h]
  _BYTE *v34; // [rsp+8h] [rbp-268h]
  const char *s2; // [rsp+18h] [rbp-258h]
  char v36; // [rsp+20h] [rbp-250h]
  _BYTE *v37; // [rsp+20h] [rbp-250h]
  _BYTE *v38; // [rsp+20h] [rbp-250h]
  _BYTE *v39; // [rsp+20h] [rbp-250h]
  _QWORD *v40; // [rsp+20h] [rbp-250h]
  char **v41; // [rsp+28h] [rbp-248h]
  unsigned __int64 v42; // [rsp+38h] [rbp-238h] BYREF
  _BYTE v43[8]; // [rsp+40h] [rbp-230h] BYREF
  char *v44; // [rsp+48h] [rbp-228h]
  char v45; // [rsp+58h] [rbp-218h] BYREF
  void *s1; // [rsp+D8h] [rbp-198h]
  __int64 v47; // [rsp+E0h] [rbp-190h]
  __int64 v48; // [rsp+E8h] [rbp-188h] BYREF
  __int64 *v49; // [rsp+F8h] [rbp-178h]
  __int64 v50; // [rsp+108h] [rbp-168h] BYREF
  char v51; // [rsp+120h] [rbp-150h]
  char **v52; // [rsp+130h] [rbp-140h] BYREF
  __int64 v53; // [rsp+138h] [rbp-138h]
  _BYTE v54[304]; // [rsp+140h] [rbp-130h] BYREF

  v3 = *(_QWORD *)(a1 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v3 + 24) )
    return;
  v5 = (unsigned __int64)&v52;
  s2 = sub_BD5D20(v3);
  v7 = v6;
  v52 = (char **)v54;
  v53 = 0x800000000LL;
  sub_C0B8E0(a1, (__int64)&v52);
  if ( !(_DWORD)v53 )
  {
    v17 = v52;
    if ( v52 == (char **)v54 )
      return;
LABEL_25:
    _libc_free(v17, v5);
    return;
  }
  v41 = &v52[4 * (unsigned int)v53];
  v8 = a2;
  v9 = a1;
  v10 = v52;
  v11 = v8;
  do
  {
    v5 = (unsigned __int64)*v10;
    sub_C0A940((__int64)v43, *v10, v10[1], *(_QWORD *)(v9 + 80));
    v13 = v51;
    if ( v51 )
    {
      if ( v47 == v7 )
      {
        v36 = v51;
        if ( !v7 || (v5 = (unsigned __int64)s2, v14 = memcmp(s1, s2, v7), v13 = v36, !v14) )
        {
          v18 = v11[2];
          v5 = v11[3];
          v19 = v43;
          v20 = *(_QWORD *)v11;
          v21 = v11[2];
          if ( v18 + 1 > v5 )
          {
            if ( v20 > (unsigned __int64)v43 || (unsigned __int64)v43 >= v20 + 224 * v18 )
            {
              v5 = sub_C8D7D0((__int64)v11, (__int64)(v11 + 4), v18 + 1, 0xE0u, &v42, v12);
              sub_D39280(v11, (const __m128i *)v5);
              v20 = v5;
              if ( v11 + 4 == *(unsigned int **)v11 )
              {
                v18 = v11[2];
                v11[3] = v42;
                v19 = v43;
                *(_QWORD *)v11 = v5;
                v21 = v18;
              }
              else
              {
                v33 = v42;
                _libc_free(*(_QWORD *)v11, v5);
                v18 = v11[2];
                v19 = v43;
                v20 = v5;
                v11[3] = v33;
                v21 = v18;
                *(_QWORD *)v11 = v5;
              }
            }
            else
            {
              v39 = &v43[-v20];
              v5 = sub_C8D7D0((__int64)v11, (__int64)(v11 + 4), v18 + 1, 0xE0u, &v42, v12);
              sub_D39280(v11, (const __m128i *)v5);
              v27 = v42;
              v20 = v5;
              if ( v11 + 4 == *(unsigned int **)v11 )
              {
                *(_QWORD *)v11 = v5;
                v11[3] = v27;
              }
              else
              {
                v29 = v42;
                _libc_free(*(_QWORD *)v11, v5);
                v20 = v5;
                *(_QWORD *)v11 = v5;
                v11[3] = v29;
              }
              v18 = v11[2];
              v19 = &v39[v20];
              v21 = v11[2];
            }
          }
          v22 = (_QWORD *)(v20 + 224 * v18);
          if ( v22 )
          {
            v23 = *(_QWORD *)v19;
            v24 = v22 + 3;
            v22[1] = v22 + 3;
            *v22 = v23;
            v22[2] = 0x800000000LL;
            v25 = *((_DWORD *)v19 + 4);
            if ( v25 && v22 + 1 != (_QWORD *)(v19 + 8) )
            {
              v26 = 16LL * v25;
              if ( v25 <= 8 )
                goto LABEL_35;
              v30 = *((_DWORD *)v19 + 4);
              v34 = v19;
              v40 = v22;
              sub_C8D5F0((__int64)(v22 + 1), v22 + 3, v25, 0x10u, (__int64)v22, (__int64)(v22 + 1));
              v19 = v34;
              v22 = v40;
              v25 = v30;
              v24 = (void *)v40[1];
              v26 = 16LL * *((unsigned int *)v34 + 4);
              if ( v26 )
              {
LABEL_35:
                v28 = v25;
                v32 = v22;
                v38 = v19;
                memcpy(v24, *((const void **)v19 + 1), v26);
                v25 = v28;
                v22 = v32;
                v19 = v38;
              }
              *((_DWORD *)v22 + 4) = v25;
            }
            v31 = v22;
            v22[19] = v22 + 21;
            v37 = v19;
            sub_D32550(v22 + 19, *((_BYTE **)v19 + 19), *((_QWORD *)v19 + 19) + *((_QWORD *)v19 + 20));
            v31[23] = v31 + 25;
            v5 = *((_QWORD *)v37 + 23);
            sub_D32550(v31 + 23, (_BYTE *)v5, v5 + *((_QWORD *)v37 + 24));
            *((_DWORD *)v31 + 54) = *((_DWORD *)v37 + 54);
            v21 = v11[2];
          }
          v13 = v51;
          v11[2] = v21 + 1;
        }
      }
      if ( v13 )
      {
        v51 = 0;
        if ( v49 != &v50 )
        {
          v5 = v50 + 1;
          j_j___libc_free_0(v49, v50 + 1);
        }
        if ( s1 != &v48 )
        {
          v5 = v48 + 1;
          j_j___libc_free_0(s1, v48 + 1);
        }
        if ( v44 != &v45 )
          _libc_free(v44, v5);
      }
    }
    v10 += 4;
  }
  while ( v41 != v10 );
  v15 = v52;
  v16 = &v52[4 * (unsigned int)v53];
  if ( v52 != v16 )
  {
    do
    {
      v16 -= 4;
      if ( *v16 != (char *)(v16 + 2) )
      {
        v5 = (unsigned __int64)(v16[2] + 1);
        j_j___libc_free_0(*v16, v5);
      }
    }
    while ( v15 != v16 );
    v15 = v52;
  }
  v17 = v15;
  if ( v15 != (char **)v54 )
    goto LABEL_25;
}
