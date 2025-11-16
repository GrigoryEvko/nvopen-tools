// Function: sub_2366420
// Address: 0x2366420
//
__int64 __fastcall sub_2366420(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v19; // [rsp+8h] [rbp-138h] BYREF
  char v20[8]; // [rsp+10h] [rbp-130h] BYREF
  __int64 v21; // [rsp+18h] [rbp-128h]
  unsigned int v22; // [rsp+28h] [rbp-118h]
  unsigned __int64 v23; // [rsp+30h] [rbp-110h]
  unsigned int v24; // [rsp+38h] [rbp-108h]
  char v25; // [rsp+40h] [rbp-100h] BYREF
  __int64 v26; // [rsp+48h] [rbp-F8h]
  unsigned int v27; // [rsp+58h] [rbp-E8h]
  char *v28; // [rsp+60h] [rbp-E0h]
  char v29; // [rsp+70h] [rbp-D0h] BYREF
  unsigned __int64 v30; // [rsp+C0h] [rbp-80h]
  __int64 v31; // [rsp+E8h] [rbp-58h]
  unsigned int v32; // [rsp+F8h] [rbp-48h]
  char *v33; // [rsp+100h] [rbp-40h]
  char v34; // [rsp+110h] [rbp-30h] BYREF

  sub_2365C20((__int64)v20, a2, a3, a4, a5, a6);
  v6 = (_QWORD *)sub_22077B0(0x110u);
  v11 = (unsigned __int64)v6;
  if ( v6 )
  {
    *v6 = &unk_4A0F6B8;
    sub_2365C20((__int64)(v6 + 1), (__int64)v20, v7, v8, v9, v10);
  }
  v19 = v11;
  sub_2353900(a1, &v19);
  if ( v19 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v19 + 8LL))(v19);
  if ( v33 != &v34 )
    _libc_free((unsigned __int64)v33);
  sub_C7D6A0(v31, 16LL * v32, 8);
  v12 = v30;
  while ( v12 )
  {
    sub_23082A0(*(_QWORD *)(v12 + 24));
    v13 = v12;
    v12 = *(_QWORD *)(v12 + 16);
    j_j___libc_free_0(v13);
  }
  if ( v28 != &v29 )
    _libc_free((unsigned __int64)v28);
  sub_C7D6A0(v26, 8LL * v27, 8);
  v14 = v23;
  v15 = v23 + 40LL * v24;
  if ( v23 != v15 )
  {
    do
    {
      v15 -= 40LL;
      if ( *(_DWORD *)(v15 + 32) > 0x40u )
      {
        v16 = *(_QWORD *)(v15 + 24);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      if ( *(_DWORD *)(v15 + 16) > 0x40u )
      {
        v17 = *(_QWORD *)(v15 + 8);
        if ( v17 )
          j_j___libc_free_0_0(v17);
      }
    }
    while ( v14 != v15 );
    v15 = v23;
  }
  if ( (char *)v15 != &v25 )
    _libc_free(v15);
  return sub_C7D6A0(v21, 16LL * v22, 8);
}
