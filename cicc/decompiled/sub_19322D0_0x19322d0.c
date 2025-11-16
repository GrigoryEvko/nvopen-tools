// Function: sub_19322D0
// Address: 0x19322d0
//
__int64 __fastcall sub_19322D0(__int64 a1, __int64 a2, const void **a3, __int64 a4, int a5, int a6)
{
  char v8; // al
  int v9; // r8d
  const void **v10; // r14
  __int64 v11; // rcx
  char v12; // dl
  __int64 v13; // rax
  unsigned int v15; // esi
  int v16; // eax
  int v17; // eax
  int v18; // r9d
  __int64 v19; // rcx
  __int64 v20; // rdx
  _BYTE *v21; // rcx
  _BYTE *v22; // r8
  size_t v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // r8d
  int v27; // r9d
  size_t v28; // rdx
  int v29; // eax
  int v30; // eax
  __int64 v31; // rcx
  int v32; // r8d
  int v33; // r9d
  _BYTE *v34; // [rsp+10h] [rbp-A0h]
  _BYTE *v35; // [rsp+10h] [rbp-A0h]
  void *s2; // [rsp+20h] [rbp-90h] BYREF
  __int64 v37; // [rsp+28h] [rbp-88h]
  _BYTE v38[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v39; // [rsp+50h] [rbp-60h] BYREF
  __int64 v40; // [rsp+58h] [rbp-58h]
  _BYTE v41[80]; // [rsp+60h] [rbp-50h] BYREF

  v8 = sub_1931280(a2, a3, (__int64 *)&s2, a4, a5, a6);
  v10 = (const void **)s2;
  if ( v8 )
  {
    v11 = *(_QWORD *)a2;
    v12 = 0;
    v13 = *(_QWORD *)(a2 + 8) + 96LL * *(unsigned int *)(a2 + 24);
    goto LABEL_3;
  }
  v15 = *(_DWORD *)(a2 + 24);
  v16 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v17 = v16 + 1;
  v18 = 2 * v15;
  if ( 4 * v17 >= 3 * v15 )
  {
    v15 *= 2;
  }
  else
  {
    v19 = v15 >> 3;
    if ( v15 - *(_DWORD *)(a2 + 20) - v17 > (unsigned int)v19 )
      goto LABEL_6;
  }
  sub_1931830(a2, v15);
  sub_1931280(a2, a3, (__int64 *)&s2, v31, v32, v33);
  v10 = (const void **)s2;
  v17 = *(_DWORD *)(a2 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a2 + 16) = v17;
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
  v20 = (unsigned int)qword_4FAF3C8;
  s2 = v38;
  v37 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3C8 )
    sub_192DAF0((__int64)&s2, (__int64)&qword_4FAF3C0, (unsigned int)qword_4FAF3C8, v19, v9, v18);
  v21 = v41;
  v40 = 0x400000000LL;
  v22 = v41;
  v39 = v41;
  if ( (_DWORD)qword_4FAF3F8 )
  {
    sub_192DA10((__int64)&v39, (__int64)&qword_4FAF3F0, v20, (__int64)v41, (int)v41, v18);
    v22 = v39;
    v21 = v41;
  }
  v23 = *((unsigned int *)v10 + 2);
  if ( v23 != (unsigned int)v37 )
    goto LABEL_12;
  v28 = 8 * v23;
  if ( v28 )
  {
    v34 = v22;
    v29 = memcmp(*v10, s2, v28);
    v22 = v34;
    v21 = v41;
    if ( v29 )
      goto LABEL_12;
  }
  v23 = *((unsigned int *)v10 + 14);
  if ( v23 != (unsigned int)v40
    || (v23 *= 8LL) != 0 && (v35 = v22, v30 = memcmp(v10[6], v22, v23), v22 = v35, v21 = v41, v30) )
  {
LABEL_12:
    --*(_DWORD *)(a2 + 20);
  }
  if ( v22 != v41 )
    _libc_free((unsigned __int64)v22);
  if ( s2 != v38 )
    _libc_free((unsigned __int64)s2);
  sub_192DAF0((__int64)v10, (__int64)a3, v23, (__int64)v21, (int)v22, v18);
  sub_192DA10((__int64)(v10 + 6), (__int64)(a3 + 6), v24, v25, v26, v27);
  v11 = *(_QWORD *)a2;
  v12 = 1;
  v13 = *(_QWORD *)(a2 + 8) + 96LL * *(unsigned int *)(a2 + 24);
LABEL_3:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 24) = v13;
  *(_QWORD *)(a1 + 8) = v11;
  *(_BYTE *)(a1 + 32) = v12;
  return a1;
}
