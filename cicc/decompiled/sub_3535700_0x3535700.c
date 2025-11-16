// Function: sub_3535700
// Address: 0x3535700
//
unsigned __int64 __fastcall sub_3535700(
        unsigned __int64 *a1,
        int *a2,
        _DWORD *a3,
        _DWORD *a4,
        _QWORD *a5,
        _QWORD *a6,
        __int64 *a7,
        __int64 *a8,
        unsigned int *a9)
{
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rcx
  int *v15; // r10
  int *v16; // r15
  bool v17; // cf
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rcx
  int v22; // edx
  int v23; // r11d
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned int v28; // eax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // r12
  int *v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rdx
  int v34; // eax
  int v35; // eax
  int v36; // eax
  __int64 v37; // rdx
  unsigned __int64 i; // r15
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  _QWORD *v44; // [rsp+0h] [rbp-70h]
  _QWORD *v45; // [rsp+8h] [rbp-68h]
  _DWORD *v46; // [rsp+10h] [rbp-60h]
  _DWORD *v47; // [rsp+18h] [rbp-58h]
  unsigned __int64 v48; // [rsp+20h] [rbp-50h]
  unsigned __int64 v49; // [rsp+28h] [rbp-48h]
  unsigned __int64 v50; // [rsp+30h] [rbp-40h]
  unsigned __int64 v51; // [rsp+38h] [rbp-38h]

  v11 = a1[1];
  v12 = *a1;
  v13 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v11 - *a1) >> 5);
  if ( v13 == 0x92492492492492LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v14 = 1;
  v15 = a2;
  v16 = a2;
  if ( v13 )
    v14 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v11 - v12) >> 5);
  v17 = __CFADD__(v14, v13);
  v18 = v14 + v13;
  v19 = (__int64)a2 - v12;
  if ( v17 )
  {
    v42 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v18 )
    {
      v50 = 0;
      v20 = 224;
      v51 = 0;
      goto LABEL_7;
    }
    if ( v18 > 0x92492492492492LL )
      v18 = 0x92492492492492LL;
    v42 = 224 * v18;
  }
  v44 = a6;
  v45 = a5;
  v46 = a4;
  v47 = a3;
  v43 = sub_22077B0(v42);
  v19 = (__int64)a2 - v12;
  v15 = a2;
  a3 = v47;
  a4 = v46;
  v51 = v43;
  a5 = v45;
  a6 = v44;
  v50 = v43 + v42;
  v20 = v43 + 224;
LABEL_7:
  v21 = v51 + v19;
  if ( v21 )
  {
    v22 = *a3;
    *(_DWORD *)(v21 + 32) = 0;
    v23 = *a4;
    *(_QWORD *)(v21 + 40) = 0;
    v24 = *a7;
    *(_DWORD *)v21 = v22;
    v25 = *a5;
    *(_DWORD *)(v21 + 4) = v23;
    v26 = *a8;
    *(_QWORD *)(v21 + 24) = v24;
    *(_QWORD *)(v21 + 8) = v25;
    v27 = *a6;
    v28 = *a9;
    *(_QWORD *)(v21 + 48) = v21 + 64;
    *(_QWORD *)(v21 + 128) = v21 + 144;
    a5 = 0;
    *(_QWORD *)(v21 + 16) = v27;
    *(_QWORD *)(v21 + 56) = 0x600000000LL;
    *(_DWORD *)(v21 + 112) = 0;
    *(_QWORD *)(v21 + 120) = 0;
    *(_QWORD *)(v21 + 136) = 0x600000000LL;
    *(_DWORD *)(v21 + 192) = 0;
    *(_WORD *)(v21 + 200) = 0;
    *(_QWORD *)(v21 + 204) = (unsigned int)v26;
    *(_QWORD *)(v21 + 212) = v28;
  }
  if ( v15 != (int *)v12 )
  {
    v49 = v12;
    v29 = v51;
    v48 = v11;
    v30 = v12;
    v31 = v15;
    while ( 1 )
    {
      if ( v29 )
      {
        *(_DWORD *)v29 = *(_DWORD *)v30;
        *(_DWORD *)(v29 + 4) = *(_DWORD *)(v30 + 4);
        *(_QWORD *)(v29 + 8) = *(_QWORD *)(v30 + 8);
        *(_QWORD *)(v29 + 16) = *(_QWORD *)(v30 + 16);
        *(_QWORD *)(v29 + 24) = *(_QWORD *)(v30 + 24);
        *(_DWORD *)(v29 + 32) = *(_DWORD *)(v30 + 32);
        v33 = *(_QWORD *)(v30 + 40);
        *(_DWORD *)(v29 + 56) = 0;
        *(_QWORD *)(v29 + 40) = v33;
        *(_QWORD *)(v29 + 48) = v29 + 64;
        *(_DWORD *)(v29 + 60) = 6;
        if ( *(_DWORD *)(v30 + 56) )
          sub_3532480(v29 + 48, v30 + 48, v29 + 64, v21, (__int64)a5, (__int64)a6);
        *(_DWORD *)(v29 + 112) = *(_DWORD *)(v30 + 112);
        v32 = *(_QWORD *)(v30 + 120);
        *(_DWORD *)(v29 + 136) = 0;
        *(_QWORD *)(v29 + 120) = v32;
        *(_QWORD *)(v29 + 128) = v29 + 144;
        *(_DWORD *)(v29 + 140) = 6;
        v21 = *(unsigned int *)(v30 + 136);
        if ( (_DWORD)v21 )
          sub_3532480(v29 + 128, v30 + 128, v29 + 144, v21, (__int64)a5, (__int64)a6);
        *(_DWORD *)(v29 + 192) = *(_DWORD *)(v30 + 192);
        *(_BYTE *)(v29 + 200) = *(_BYTE *)(v30 + 200);
        *(_BYTE *)(v29 + 201) = *(_BYTE *)(v30 + 201);
        *(_DWORD *)(v29 + 204) = *(_DWORD *)(v30 + 204);
        *(_DWORD *)(v29 + 208) = *(_DWORD *)(v30 + 208);
        *(_DWORD *)(v29 + 212) = *(_DWORD *)(v30 + 212);
        *(_DWORD *)(v29 + 216) = *(_DWORD *)(v30 + 216);
      }
      v30 += 224LL;
      if ( v31 == (int *)v30 )
        break;
      v29 += 224LL;
    }
    v15 = v31;
    v11 = v48;
    v12 = v49;
    v20 = v29 + 448;
  }
  if ( v15 != (int *)v11 )
  {
    do
    {
      v36 = *v16;
      v37 = (unsigned int)v16[14];
      *(_DWORD *)(v20 + 56) = 0;
      *(_DWORD *)(v20 + 60) = 6;
      *(_DWORD *)v20 = v36;
      *(_DWORD *)(v20 + 4) = v16[1];
      *(_QWORD *)(v20 + 8) = *((_QWORD *)v16 + 1);
      *(_QWORD *)(v20 + 16) = *((_QWORD *)v16 + 2);
      *(_QWORD *)(v20 + 24) = *((_QWORD *)v16 + 3);
      *(_DWORD *)(v20 + 32) = v16[8];
      *(_QWORD *)(v20 + 40) = *((_QWORD *)v16 + 5);
      *(_QWORD *)(v20 + 48) = v20 + 64;
      if ( (_DWORD)v37 )
        sub_3532480(v20 + 48, (__int64)(v16 + 12), v37, v21, (__int64)a5, (__int64)a6);
      *(_DWORD *)(v20 + 136) = 0;
      v34 = v16[28];
      *(_DWORD *)(v20 + 140) = 6;
      *(_DWORD *)(v20 + 112) = v34;
      *(_QWORD *)(v20 + 120) = *((_QWORD *)v16 + 15);
      *(_QWORD *)(v20 + 128) = v20 + 144;
      if ( v16[34] )
        sub_3532480(v20 + 128, (__int64)(v16 + 32), v37, v21, (__int64)a5, (__int64)a6);
      v35 = v16[48];
      v16 += 56;
      v20 += 224;
      *(_DWORD *)(v20 - 32) = v35;
      *(_BYTE *)(v20 - 24) = *((_BYTE *)v16 - 24);
      *(_BYTE *)(v20 - 23) = *((_BYTE *)v16 - 23);
      *(_DWORD *)(v20 - 20) = *(v16 - 5);
      *(_DWORD *)(v20 - 16) = *(v16 - 4);
      *(_DWORD *)(v20 - 12) = *(v16 - 3);
      *(_DWORD *)(v20 - 8) = *(v16 - 2);
    }
    while ( (int *)v11 != v16 );
  }
  for ( i = v12; i != v11; i += 224LL )
  {
    v39 = *(_QWORD *)(i + 128);
    if ( v39 != i + 144 )
      _libc_free(v39);
    v40 = *(_QWORD *)(i + 48);
    if ( v40 != i + 64 )
      _libc_free(v40);
  }
  if ( v12 )
    j_j___libc_free_0(v12);
  a1[1] = v20;
  *a1 = v51;
  a1[2] = v50;
  return v50;
}
