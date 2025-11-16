// Function: sub_2366110
// Address: 0x2366110
//
__int64 __fastcall sub_2366110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rbx
  char *v14; // rdi
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  char v22[8]; // [rsp+0h] [rbp-130h] BYREF
  __int64 v23; // [rsp+8h] [rbp-128h]
  unsigned int v24; // [rsp+18h] [rbp-118h]
  unsigned __int64 v25; // [rsp+20h] [rbp-110h]
  unsigned int v26; // [rsp+28h] [rbp-108h]
  char v27; // [rsp+30h] [rbp-100h] BYREF
  __int64 v28; // [rsp+38h] [rbp-F8h]
  unsigned int v29; // [rsp+48h] [rbp-E8h]
  char *v30; // [rsp+50h] [rbp-E0h]
  char v31; // [rsp+60h] [rbp-D0h] BYREF
  unsigned __int64 v32; // [rsp+B0h] [rbp-80h]
  __int64 v33; // [rsp+D8h] [rbp-58h]
  unsigned int v34; // [rsp+E8h] [rbp-48h]
  char *v35; // [rsp+F0h] [rbp-40h]
  char v36; // [rsp+100h] [rbp-30h] BYREF

  v7 = a3;
  sub_2365C20((__int64)v22, a2, a3, a4, a5, a6);
  v8 = (_QWORD *)sub_22077B0(0x110u);
  v13 = v8;
  if ( v8 )
  {
    *v8 = &unk_4A0F6B8;
    sub_2365C20((__int64)(v8 + 1), (__int64)v22, v9, v10, v11, v12);
  }
  v14 = v35;
  *(_QWORD *)a1 = v13;
  *(_BYTE *)(a1 + 8) = v7;
  if ( v14 != &v36 )
    _libc_free((unsigned __int64)v14);
  sub_C7D6A0(v33, 16LL * v34, 8);
  v15 = v32;
  while ( v15 )
  {
    sub_23082A0(*(_QWORD *)(v15 + 24));
    v16 = v15;
    v15 = *(_QWORD *)(v15 + 16);
    j_j___libc_free_0(v16);
  }
  if ( v30 != &v31 )
    _libc_free((unsigned __int64)v30);
  sub_C7D6A0(v28, 8LL * v29, 8);
  v17 = v25;
  v18 = v25 + 40LL * v26;
  if ( v25 != v18 )
  {
    do
    {
      v18 -= 40LL;
      if ( *(_DWORD *)(v18 + 32) > 0x40u )
      {
        v19 = *(_QWORD *)(v18 + 24);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
      if ( *(_DWORD *)(v18 + 16) > 0x40u )
      {
        v20 = *(_QWORD *)(v18 + 8);
        if ( v20 )
          j_j___libc_free_0_0(v20);
      }
    }
    while ( v17 != v18 );
    v18 = v25;
  }
  if ( (char *)v18 != &v27 )
    _libc_free(v18);
  sub_C7D6A0(v23, 16LL * v24, 8);
  return a1;
}
