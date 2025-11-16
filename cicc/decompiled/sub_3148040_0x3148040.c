// Function: sub_3148040
// Address: 0x3148040
//
__int64 __fastcall sub_3148040(_QWORD *a1, char a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  char v12; // [rsp+8h] [rbp-68h]
  _BYTE v13[16]; // [rsp+10h] [rbp-60h] BYREF
  void (__fastcall *v14)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-50h]
  unsigned __int64 v15; // [rsp+30h] [rbp-40h]
  unsigned __int64 v16; // [rsp+38h] [rbp-38h]
  __int64 v17; // [rsp+40h] [rbp-30h]
  __int64 v18; // [rsp+48h] [rbp-28h]
  __int64 v19; // [rsp+50h] [rbp-20h]
  unsigned int v20; // [rsp+58h] [rbp-18h]

  v11 = 4;
  v12 = a2;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  if ( !sub_B2FC80((__int64)a1) )
    sub_3146A80(&v11, a1, v2, v3, v4, v5);
  v6 = v11;
  sub_C7D6A0(v18, 16LL * v20, 8);
  v7 = v16;
  if ( v16 )
  {
    sub_C7D6A0(*(_QWORD *)(v16 + 8), 16LL * *(unsigned int *)(v16 + 24), 8);
    j_j___libc_free_0(v7);
  }
  v8 = v15;
  if ( v15 )
  {
    v9 = *(_QWORD *)(v15 + 32);
    if ( v9 != v15 + 48 )
      _libc_free(v9);
    sub_C7D6A0(*(_QWORD *)(v8 + 8), 8LL * *(unsigned int *)(v8 + 24), 4);
    j_j___libc_free_0(v8);
  }
  if ( v14 )
    v14(v13, v13, 3);
  return v6;
}
