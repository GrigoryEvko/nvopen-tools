// Function: sub_192F350
// Address: 0x192f350
//
void __fastcall sub_192F350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 *v8; // r15
  unsigned __int64 *i; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11[2]; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE v12[32]; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int64 v13[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v14[32]; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v15[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v16[32]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v17[2]; // [rsp+A0h] [rbp-60h] BYREF
  _BYTE v18[80]; // [rsp+B0h] [rbp-50h] BYREF

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
  v11[0] = (unsigned __int64)v12;
  v11[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3C8 )
    sub_192DAF0((__int64)v11, (__int64)&qword_4FAF3C0, a3, a4, a5, a6);
  v6 = (unsigned int)qword_4FAF3F8;
  v13[0] = (unsigned __int64)v14;
  v13[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3F8 )
    sub_192DA10((__int64)v13, (__int64)&qword_4FAF3F0, a3, (unsigned int)qword_4FAF3F8, a5, a6);
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
  v7 = (unsigned int)qword_4FAF348;
  v15[0] = (unsigned __int64)v16;
  v15[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FAF348 )
    sub_192DAF0((__int64)v15, (__int64)&qword_4FAF340, (unsigned int)qword_4FAF348, v6, a5, a6);
  v17[0] = (unsigned __int64)v18;
  v17[1] = 0x400000000LL;
  if ( !(_DWORD)qword_4FAF378 )
  {
    v8 = *(unsigned __int64 **)(a1 + 8);
    i = &v8[12 * *(unsigned int *)(a1 + 24)];
    if ( v8 == i )
      goto LABEL_18;
    goto LABEL_11;
  }
  sub_192DA10((__int64)v17, (__int64)&qword_4FAF370, v7, v6, a5, a6);
  v8 = *(unsigned __int64 **)(a1 + 8);
  for ( i = &v8[12 * *(unsigned int *)(a1 + 24)]; v8 != i; v8 += 12 )
  {
LABEL_11:
    v10 = v8[6];
    if ( (unsigned __int64 *)v10 != v8 + 8 )
      _libc_free(v10);
    if ( (unsigned __int64 *)*v8 != v8 + 2 )
      _libc_free(*v8);
  }
  if ( (_BYTE *)v17[0] != v18 )
    _libc_free(v17[0]);
LABEL_18:
  if ( (_BYTE *)v15[0] != v16 )
    _libc_free(v15[0]);
  if ( (_BYTE *)v13[0] != v14 )
    _libc_free(v13[0]);
  if ( (_BYTE *)v11[0] != v12 )
    _libc_free(v11[0]);
}
