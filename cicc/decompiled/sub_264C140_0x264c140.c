// Function: sub_264C140
// Address: 0x264c140
//
__int64 __fastcall sub_264C140(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  void *v11; // r8
  __int64 v12; // r15
  unsigned __int64 v13; // r10
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r9
  unsigned __int64 v17; // r11
  _DWORD *v18; // rax
  __int64 v19; // r12
  _DWORD *v20; // rsi
  __int64 v21; // rdi
  _DWORD *v22; // rdx
  char *v23; // rbx
  unsigned __int64 v24; // rcx
  char *v25; // rbx
  unsigned __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // r8
  int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rdi
  _DWORD *v36; // rax
  _DWORD *v37; // rax
  _DWORD *v38; // rax
  int v39; // edx
  char *v40; // rbx
  size_t n; // [rsp+8h] [rbp-98h]
  void *src; // [rsp+10h] [rbp-90h]
  int v44; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 v45; // [rsp+28h] [rbp-78h] BYREF
  _BYTE *v46; // [rsp+30h] [rbp-70h] BYREF
  char *v47; // [rsp+38h] [rbp-68h]
  char *v48; // [rsp+40h] [rbp-60h]
  char *v49; // [rsp+48h] [rbp-58h]
  int v50; // [rsp+50h] [rbp-50h]
  __int64 v51; // [rsp+58h] [rbp-48h]
  unsigned __int64 v52; // [rsp+60h] [rbp-40h]

  v9 = sub_FFE1E0(*(_QWORD **)(a1 + 32), a2, &v45, &v44);
  if ( !v10 || (v11 = (void *)v9, v12 = *a5, v13 = 16 * v10, v14 = 16 * v10 + v9, v11 == (void *)v14) )
  {
    LODWORD(v19) = 0;
    return (unsigned int)v19;
  }
  v15 = v12 + 136;
  v16 = 0;
  v17 = v12
      + 8 * (((unsigned __int64)(v14 - (_QWORD)v11 - 16) >> 4) + ((v14 - (_QWORD)v11 - 16) & 0xFFFFFFFFFFFFFFF0LL))
      + 272;
  do
  {
    *a5 = v15;
    v18 = *(_DWORD **)(v15 - 128);
    v19 = *(unsigned int *)(v15 - 120);
    v20 = &v18[v19];
    v21 = (4 * v19) >> 2;
    if ( (4 * v19) >> 4 )
    {
      v22 = &v18[4 * ((4 * v19) >> 4)];
      while ( !*v18 )
      {
        if ( v18[1] )
        {
          v36 = v18 + 1;
          LOBYTE(v36) = v20 != v36;
          v16 = (unsigned int)v36 | (unsigned int)v16;
          goto LABEL_12;
        }
        if ( v18[2] )
        {
          v37 = v18 + 2;
          LOBYTE(v37) = v20 != v37;
          v16 = (unsigned int)v37 | (unsigned int)v16;
          goto LABEL_12;
        }
        if ( v18[3] )
        {
          v38 = v18 + 3;
          LOBYTE(v38) = v20 != v38;
          v16 = (unsigned int)v38 | (unsigned int)v16;
          goto LABEL_12;
        }
        v18 += 4;
        if ( v22 == v18 )
        {
          v21 = v20 - v18;
          goto LABEL_24;
        }
      }
LABEL_11:
      LOBYTE(v18) = v20 != v18;
      v16 = (unsigned int)v18 | (unsigned int)v16;
      goto LABEL_12;
    }
LABEL_24:
    if ( v21 != 2 )
    {
      if ( v21 != 3 )
      {
        if ( v21 == 1 && *v18 )
          goto LABEL_11;
        goto LABEL_12;
      }
      if ( *v18 )
        goto LABEL_11;
      ++v18;
    }
    if ( *v18 )
      goto LABEL_11;
    v39 = v18[1];
    ++v18;
    if ( v39 )
      goto LABEL_11;
LABEL_12:
    v15 += 136;
  }
  while ( v15 != v17 );
  if ( (_BYTE)v16 )
  {
    v46 = a2;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    if ( v13 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    src = v11;
    v23 = 0;
    if ( v13 )
    {
      n = v13;
      v47 = (char *)sub_22077B0(v13);
      v23 = &v47[n];
      v49 = &v47[n];
      memcpy(v47, src, n);
    }
    v24 = *(unsigned int *)(a6 + 12);
    v48 = v23;
    v25 = (char *)&v46;
    v26 = *(_QWORD *)a6;
    v50 = v44;
    v51 = v45;
    v27 = *(unsigned int *)(a6 + 8);
    v28 = v27 + 1;
    v52 = 0xF0F0F0F0F0F0F0F1LL * ((v12 - a3) >> 3);
    v29 = v27;
    if ( v27 + 1 > v24 )
    {
      if ( v26 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v26 + 56 * v27 )
      {
        sub_264C020(a6, v27 + 1, v27, v24, v28, v16);
        v27 = *(unsigned int *)(a6 + 8);
        v26 = *(_QWORD *)a6;
        v29 = *(_DWORD *)(a6 + 8);
      }
      else
      {
        v40 = (char *)&v46 - v26;
        sub_264C020(a6, v27 + 1, v27, v24, v28, v16);
        v26 = *(_QWORD *)a6;
        v27 = *(unsigned int *)(a6 + 8);
        v25 = &v40[*(_QWORD *)a6];
        v29 = *(_DWORD *)(a6 + 8);
      }
    }
    v30 = v26 + 56 * v27;
    if ( v30 )
    {
      *(_QWORD *)v30 = *(_QWORD *)v25;
      v31 = *((_QWORD *)v25 + 1);
      *((_QWORD *)v25 + 1) = 0;
      *(_QWORD *)(v30 + 8) = v31;
      v32 = *((_QWORD *)v25 + 2);
      *((_QWORD *)v25 + 2) = 0;
      *(_QWORD *)(v30 + 16) = v32;
      v33 = *((_QWORD *)v25 + 3);
      *((_QWORD *)v25 + 3) = 0;
      *(_QWORD *)(v30 + 24) = v33;
      *(_DWORD *)(v30 + 32) = *((_DWORD *)v25 + 8);
      *(_QWORD *)(v30 + 40) = *((_QWORD *)v25 + 5);
      *(_QWORD *)(v30 + 48) = *((_QWORD *)v25 + 6);
      v29 = *(_DWORD *)(a6 + 8);
    }
    v34 = (unsigned __int64)v47;
    *(_DWORD *)(a6 + 8) = v29 + 1;
    if ( v34 )
      j_j___libc_free_0(v34);
  }
  return (unsigned int)v19;
}
