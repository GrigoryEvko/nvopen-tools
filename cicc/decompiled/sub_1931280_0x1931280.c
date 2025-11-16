// Function: sub_1931280
// Address: 0x1931280
//
__int64 __fastcall sub_1931280(__int64 a1, const void **a2, __int64 *a3, __int64 a4, int a5, int a6)
{
  int v7; // r13d
  unsigned int v8; // r12d
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // r9
  __int64 v14; // r10
  __int64 v15; // r12
  size_t v16; // r15
  __int64 v17; // r13
  __int64 v18; // rbx
  int v19; // eax
  char v20; // al
  int v21; // eax
  __int64 v22; // rdx
  size_t v23; // rdx
  int v24; // eax
  int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-170h]
  __int64 v27; // [rsp+0h] [rbp-170h]
  __int64 v28; // [rsp+0h] [rbp-170h]
  __int64 v29; // [rsp+0h] [rbp-170h]
  __int64 v30; // [rsp+0h] [rbp-170h]
  __int64 v31; // [rsp+8h] [rbp-168h]
  __int64 v32; // [rsp+8h] [rbp-168h]
  __int64 v33; // [rsp+8h] [rbp-168h]
  __int64 v34; // [rsp+8h] [rbp-168h]
  __int64 v35; // [rsp+8h] [rbp-168h]
  int v36; // [rsp+14h] [rbp-15Ch]
  void *v37; // [rsp+18h] [rbp-158h]
  void *s2; // [rsp+20h] [rbp-150h]
  __int64 v39; // [rsp+28h] [rbp-148h]
  __int64 v40; // [rsp+30h] [rbp-140h]
  __int64 n; // [rsp+38h] [rbp-138h]
  __int64 *v42; // [rsp+58h] [rbp-118h]
  __int64 v43; // [rsp+70h] [rbp-100h]
  int v44; // [rsp+78h] [rbp-F8h]
  unsigned int v45; // [rsp+7Ch] [rbp-F4h]
  _BYTE *v46; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v47; // [rsp+88h] [rbp-E8h]
  _BYTE v48[32]; // [rsp+90h] [rbp-E0h] BYREF
  _BYTE *v49; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-B8h]
  _BYTE v51[32]; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int64 v52[2]; // [rsp+E0h] [rbp-90h] BYREF
  _BYTE v53[32]; // [rsp+F0h] [rbp-80h] BYREF
  unsigned __int64 v54[2]; // [rsp+110h] [rbp-60h] BYREF
  _BYTE v55[80]; // [rsp+120h] [rbp-50h] BYREF

  v7 = *(_DWORD *)(a1 + 24);
  if ( v7 )
  {
    v43 = *(_QWORD *)(a1 + 8);
    if ( !byte_4FAF3A0 && (unsigned int)sub_2207590(&byte_4FAF3A0) )
    {
      qword_4FAF3D0 = 0;
      qword_4FAF3C0 = (__int64)&qword_4FAF3D0;
      qword_4FAF3F0 = (__int64)&unk_4FAF400;
      qword_4FAF3F8 = 0x400000000LL;
      qword_4FAF3C8 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF3C0, &qword_4A427C0);
      sub_2207640(&byte_4FAF3A0);
    }
    v46 = v48;
    v47 = 0x400000000LL;
    if ( (_DWORD)qword_4FAF3C8 )
      sub_192DAF0((__int64)&v46, (__int64)&qword_4FAF3C0, (__int64)a3, a4, a5, a6);
    v10 = (unsigned int)qword_4FAF3F8;
    v49 = v51;
    v50 = 0x400000000LL;
    if ( (_DWORD)qword_4FAF3F8 )
      sub_192DA10((__int64)&v49, (__int64)&qword_4FAF3F0, (__int64)a3, (unsigned int)qword_4FAF3F8, a5, a6);
    if ( !byte_4FAF328 && (unsigned int)sub_2207590(&byte_4FAF328) )
    {
      qword_4FAF350 = 1;
      qword_4FAF340 = (__int64)&qword_4FAF350;
      qword_4FAF370 = (__int64)&unk_4FAF380;
      qword_4FAF378 = 0x400000000LL;
      qword_4FAF348 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF340, &qword_4A427C0);
      sub_2207640(&byte_4FAF328);
    }
    v11 = (unsigned int)qword_4FAF348;
    v52[0] = (unsigned __int64)v53;
    v52[1] = 0x400000000LL;
    if ( (_DWORD)qword_4FAF348 )
      sub_192DAF0((__int64)v52, (__int64)&qword_4FAF340, (unsigned int)qword_4FAF348, v10, a5, a6);
    v54[1] = 0x400000000LL;
    v54[0] = (unsigned __int64)v55;
    if ( (_DWORD)qword_4FAF378 )
      sub_192DA10((__int64)v54, (__int64)&qword_4FAF370, v11, v10, a5, a6);
    v12 = sub_1930F10(*a2, (__int64)*a2 + 8 * *((unsigned int *)a2 + 2));
    v13 = (unsigned int)v47;
    v14 = (unsigned int)v50;
    v36 = v7 - 1;
    v45 = (v7 - 1) & v12;
    v42 = a3;
    v15 = *((unsigned int *)a2 + 2);
    s2 = v46;
    v44 = 1;
    v37 = v49;
    n = 8LL * (unsigned int)v47;
    v40 = 0;
    v39 = 8LL * (unsigned int)v50;
    v16 = 8 * v15;
    while ( 1 )
    {
      v17 = v43 + 96LL * v45;
      v18 = *(unsigned int *)(v17 + 8);
      if ( v15 == v18 )
      {
        if ( !v16 || (v28 = v14, v33 = v13, v21 = memcmp(*a2, *(const void **)v17, v16), v13 = v33, v14 = v28, !v21) )
        {
          v22 = *((unsigned int *)a2 + 14);
          if ( v22 == *(_DWORD *)(v17 + 56) )
          {
            v23 = 8 * v22;
            if ( !v23
              || (v29 = v14, v34 = v13, v24 = memcmp(a2[6], *(const void **)(v17 + 48), v23), v13 = v34, v14 = v29, !v24) )
            {
              *v42 = v17;
              v8 = 1;
              goto LABEL_31;
            }
          }
        }
      }
      if ( v18 == v13 )
      {
        if ( !n || (v26 = v14, v31 = v13, v19 = memcmp(*(const void **)v17, s2, n), v13 = v31, v14 = v26, !v19) )
        {
          if ( *(_DWORD *)(v17 + 56) == v14 )
          {
            if ( !v39 )
              break;
            v30 = v14;
            v35 = v13;
            v25 = memcmp(*(const void **)(v17 + 48), v37, v39);
            v13 = v35;
            v14 = v30;
            if ( !v25 )
              break;
          }
        }
      }
      v27 = v14;
      v32 = v13;
      v20 = sub_192E340(v17, (__int64)v52);
      v13 = v32;
      v14 = v27;
      if ( !v40 )
      {
        if ( !v20 )
          v17 = 0;
        v40 = v17;
      }
      v45 = v36 & (v44 + v45);
      ++v44;
    }
    if ( v40 )
      v17 = v40;
    *v42 = v17;
    v8 = 0;
LABEL_31:
    if ( (_BYTE *)v54[0] != v55 )
      _libc_free(v54[0]);
    if ( (_BYTE *)v52[0] != v53 )
      _libc_free(v52[0]);
    if ( v49 != v51 )
      _libc_free((unsigned __int64)v49);
    if ( v46 != v48 )
      _libc_free((unsigned __int64)v46);
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v8;
}
