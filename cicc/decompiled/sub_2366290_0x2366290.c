// Function: sub_2366290
// Address: 0x2366290
//
__int64 __fastcall sub_2366290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r14
  char v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rbx
  char *v15; // rdi
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  char v23[8]; // [rsp+0h] [rbp-140h] BYREF
  __int64 v24; // [rsp+8h] [rbp-138h]
  unsigned int v25; // [rsp+18h] [rbp-128h]
  unsigned __int64 v26; // [rsp+20h] [rbp-120h]
  unsigned int v27; // [rsp+28h] [rbp-118h]
  char v28; // [rsp+30h] [rbp-110h] BYREF
  __int64 v29; // [rsp+38h] [rbp-108h]
  unsigned int v30; // [rsp+48h] [rbp-F8h]
  char *v31; // [rsp+50h] [rbp-F0h]
  char v32; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int64 v33; // [rsp+B0h] [rbp-90h]
  __int64 v34; // [rsp+D8h] [rbp-68h]
  unsigned int v35; // [rsp+E8h] [rbp-58h]
  char *v36; // [rsp+F0h] [rbp-50h]
  char v37; // [rsp+100h] [rbp-40h] BYREF

  v6 = a3;
  v8 = a4;
  sub_2365C20((__int64)v23, a2, a3, a4, a5, a6);
  v9 = (_QWORD *)sub_22077B0(0x110u);
  v14 = v9;
  if ( v9 )
  {
    *v9 = &unk_4A0F6B8;
    sub_2365C20((__int64)(v9 + 1), (__int64)v23, v10, v11, v12, v13);
  }
  v15 = v36;
  *(_QWORD *)a1 = v14;
  *(_BYTE *)(a1 + 8) = v6;
  *(_BYTE *)(a1 + 9) = v8;
  if ( v15 != &v37 )
    _libc_free((unsigned __int64)v15);
  sub_C7D6A0(v34, 16LL * v35, 8);
  v16 = v33;
  while ( v16 )
  {
    sub_23082A0(*(_QWORD *)(v16 + 24));
    v17 = v16;
    v16 = *(_QWORD *)(v16 + 16);
    j_j___libc_free_0(v17);
  }
  if ( v31 != &v32 )
    _libc_free((unsigned __int64)v31);
  sub_C7D6A0(v29, 8LL * v30, 8);
  v18 = v26;
  v19 = v26 + 40LL * v27;
  if ( v26 != v19 )
  {
    do
    {
      v19 -= 40LL;
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
      {
        v20 = *(_QWORD *)(v19 + 24);
        if ( v20 )
          j_j___libc_free_0_0(v20);
      }
      if ( *(_DWORD *)(v19 + 16) > 0x40u )
      {
        v21 = *(_QWORD *)(v19 + 8);
        if ( v21 )
          j_j___libc_free_0_0(v21);
      }
    }
    while ( v18 != v19 );
    v19 = v26;
  }
  if ( (char *)v19 != &v28 )
    _libc_free(v19);
  sub_C7D6A0(v24, 16LL * v25, 8);
  return a1;
}
