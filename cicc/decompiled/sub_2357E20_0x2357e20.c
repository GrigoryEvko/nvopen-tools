// Function: sub_2357E20
// Address: 0x2357e20
//
void __fastcall sub_2357E20(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // [rsp+8h] [rbp-2E8h] BYREF
  unsigned __int64 v13[2]; // [rsp+10h] [rbp-2E0h] BYREF
  char v14; // [rsp+20h] [rbp-2D0h] BYREF
  char *v15; // [rsp+60h] [rbp-290h]
  char v16; // [rsp+70h] [rbp-280h] BYREF
  char *v17; // [rsp+B0h] [rbp-240h]
  char v18; // [rsp+C0h] [rbp-230h] BYREF
  char *v19; // [rsp+100h] [rbp-1F0h]
  char v20; // [rsp+110h] [rbp-1E0h] BYREF
  char *v21; // [rsp+150h] [rbp-1A0h]
  char v22; // [rsp+160h] [rbp-190h] BYREF
  unsigned __int64 v23; // [rsp+1A8h] [rbp-148h]
  char v24; // [rsp+1BCh] [rbp-134h]
  __int64 v25; // [rsp+2C0h] [rbp-30h]

  sub_234AC00((__int64)v13, a2, a3, a4, a5, a6);
  v25 = *(_QWORD *)(a2 + 688);
  v6 = (_QWORD *)sub_22077B0(0x2C0u);
  v11 = (unsigned __int64)v6;
  if ( v6 )
  {
    *v6 = &unk_4A0DEB8;
    sub_234AC00((__int64)(v6 + 1), (__int64)v13, v7, v8, v9, v10);
    *(_QWORD *)(v11 + 696) = v25;
  }
  v12 = v11;
  sub_2356EF0(a1, &v12);
  if ( v12 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v12 + 8LL))(v12);
  if ( !v24 )
    _libc_free(v23);
  if ( v21 != &v22 )
    _libc_free((unsigned __int64)v21);
  if ( v19 != &v20 )
    _libc_free((unsigned __int64)v19);
  if ( v17 != &v18 )
    _libc_free((unsigned __int64)v17);
  if ( v15 != &v16 )
    _libc_free((unsigned __int64)v15);
  if ( (char *)v13[0] != &v14 )
    _libc_free(v13[0]);
}
