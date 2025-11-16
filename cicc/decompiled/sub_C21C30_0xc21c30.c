// Function: sub_C21C30
// Address: 0xc21c30
//
__int64 __fastcall sub_C21C30(__int64 a1)
{
  __int64 v1; // r12
  char *v2; // rax
  char *v3; // r13
  _BYTE *v4; // rcx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v11; // [rsp+8h] [rbp-A8h] BYREF
  char v12[8]; // [rsp+10h] [rbp-A0h] BYREF
  int v13; // [rsp+18h] [rbp-98h] BYREF
  __int64 v14; // [rsp+20h] [rbp-90h]
  int *v15; // [rsp+28h] [rbp-88h]
  int *v16; // [rsp+30h] [rbp-80h]
  __int64 v17; // [rsp+38h] [rbp-78h]
  _BYTE *v18; // [rsp+40h] [rbp-70h]
  char *v19; // [rsp+48h] [rbp-68h]
  char *v20; // [rsp+50h] [rbp-60h]
  __int64 v21; // [rsp+58h] [rbp-58h]
  __int64 v22; // [rsp+60h] [rbp-50h]
  __int64 v23; // [rsp+68h] [rbp-48h]
  __int64 v24; // [rsp+70h] [rbp-40h]
  __int64 v25; // [rsp+78h] [rbp-38h]
  __int64 v26; // [rsp+80h] [rbp-30h]
  __int64 v27; // [rsp+88h] [rbp-28h]

  v1 = 4LL * unk_497B318;
  if ( (unsigned __int64)(4LL * unk_497B318) > 0x7FFFFFFFFFFFFFFCLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v1 )
  {
    v2 = (char *)sub_22077B0(4LL * unk_497B318);
    v3 = &v2[v1];
    v4 = memcpy(v2, &unk_3F88320, 4LL * unk_497B318);
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  v18 = v4;
  v16 = &v13;
  v15 = &v13;
  v13 = 0;
  v14 = 0;
  v17 = 0;
  v19 = v3;
  v20 = v3;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  sub_EFC170(&v11, v12, a1 + 8);
  v5 = v11;
  v6 = *(_QWORD *)(a1 + 80);
  v11 = 0;
  *(_QWORD *)(a1 + 80) = v5;
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 8);
    if ( v7 )
      j_j___libc_free_0(v7, *(_QWORD *)(v6 + 24) - v7);
    j_j___libc_free_0(v6, 88);
    v8 = v11;
    if ( v11 )
    {
      v9 = *(_QWORD *)(v11 + 8);
      if ( v9 )
        j_j___libc_free_0(v9, *(_QWORD *)(v11 + 24) - v9);
      j_j___libc_free_0(v8, 88);
    }
  }
  if ( v21 )
    j_j___libc_free_0(v21, v23 - v21);
  if ( v18 )
    j_j___libc_free_0(v18, v20 - v18);
  return sub_C1F060(v14);
}
