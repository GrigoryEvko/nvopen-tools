// Function: sub_39E85A0
// Address: 0x39e85a0
//
void __fastcall sub_39E85A0(
        unsigned int a1,
        unsigned __int8 *a2,
        __int64 a3,
        unsigned __int8 *a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        char a8,
        __int64 a9)
{
  unsigned int v9; // r10d
  unsigned __int8 *v10; // r15
  __int64 v11; // r14
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  _BYTE *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // rcx
  unsigned int v25; // [rsp+28h] [rbp-168h]
  unsigned __int8 *v26; // [rsp+30h] [rbp-160h] BYREF
  __int64 v27; // [rsp+38h] [rbp-158h]
  unsigned __int8 **v28; // [rsp+40h] [rbp-150h] BYREF
  __int16 v29; // [rsp+50h] [rbp-140h]
  _BYTE v30[16]; // [rsp+60h] [rbp-130h] BYREF
  __int16 v31; // [rsp+70h] [rbp-120h]
  _BYTE v32[16]; // [rsp+80h] [rbp-110h] BYREF
  __int16 v33; // [rsp+90h] [rbp-100h]
  char *v34; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned int v35; // [rsp+A8h] [rbp-E8h]
  __int16 v36; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned __int8 *v37; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+D8h] [rbp-B8h]
  _BYTE v39[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v9 = a1;
  v10 = a2;
  v11 = a3;
  v26 = a4;
  v27 = a5;
  v37 = v39;
  v38 = 0x8000000000LL;
  if ( !a8 )
  {
    if ( a3 )
    {
      v11 = 0;
      v10 = (unsigned __int8 *)byte_3F871B3;
      v36 = 261;
      v34 = (char *)&v26;
      v22 = sub_16C4E60((__int64)&v34, 2u);
      v9 = a1;
      if ( !v22 )
      {
        LODWORD(v38) = 0;
        sub_39E8530((__int64)&v37, a2, &a2[a3], v23, (int)&v34, (int)&v37);
        v29 = 261;
        v31 = 257;
        v36 = 257;
        v33 = 257;
        v28 = &v26;
        sub_16C4D40((__int64)&v37, (__int64)&v28, (__int64)v30, (__int64)v32, (__int64)&v34);
        v9 = a1;
        v26 = v37;
        v27 = (unsigned int)v38;
      }
    }
  }
  v13 = *(_QWORD *)(a9 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a9 + 16) - v13) <= 6 )
  {
    v25 = v9;
    v21 = sub_16E7EE0(a9, "\t.file\t", 7u);
    v9 = v25;
    v14 = v21;
  }
  else
  {
    *(_DWORD *)v13 = 1768304137;
    v14 = a9;
    *(_WORD *)(v13 + 4) = 25964;
    *(_BYTE *)(v13 + 6) = 9;
    *(_QWORD *)(a9 + 24) += 7LL;
  }
  v15 = sub_16E7A90(v14, v9);
  v16 = *(_BYTE **)(v15 + 24);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
  {
    sub_16E7DE0(v15, 32);
  }
  else
  {
    *(_QWORD *)(v15 + 24) = v16 + 1;
    *v16 = 32;
  }
  if ( v11 )
  {
    sub_39E0070(v10, v11, a9);
    v19 = *(_BYTE **)(a9 + 24);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(a9 + 16) )
    {
      sub_16E7DE0(a9, 32);
    }
    else
    {
      *(_QWORD *)(a9 + 24) = v19 + 1;
      *v19 = 32;
    }
  }
  sub_39E0070(v26, v27, a9);
  if ( a6 )
  {
    v17 = *(_QWORD *)(a9 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a9 + 16) - v17) <= 6 )
    {
      v18 = sub_16E7EE0(a9, " md5 0x", 7u);
    }
    else
    {
      *(_DWORD *)v17 = 895773984;
      v18 = a9;
      *(_WORD *)(v17 + 4) = 12320;
      *(_BYTE *)(v17 + 6) = 120;
      *(_QWORD *)(a9 + 24) += 7LL;
    }
    sub_16C1C80(&v34, a6);
    sub_16E7EE0(v18, v34, v35);
    if ( v34 != (char *)&v36 )
      _libc_free((unsigned __int64)v34);
  }
  if ( *(_BYTE *)(a7 + 16) )
  {
    v20 = *(_QWORD **)(a9 + 24);
    if ( *(_QWORD *)(a9 + 16) - (_QWORD)v20 <= 7u )
    {
      sub_16E7EE0(a9, " source ", 8u);
    }
    else
    {
      *v20 = 0x20656372756F7320LL;
      *(_QWORD *)(a9 + 24) += 8LL;
    }
    sub_39E0070(*(unsigned __int8 **)a7, *(_QWORD *)(a7 + 8), a9);
  }
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
}
