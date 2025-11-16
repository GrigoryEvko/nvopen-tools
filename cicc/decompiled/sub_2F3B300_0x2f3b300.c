// Function: sub_2F3B300
// Address: 0x2f3b300
//
__int64 __fastcall sub_2F3B300(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 (*v16)(void); // rdx
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // r12
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  _QWORD v23[4]; // [rsp+0h] [rbp-180h] BYREF
  _QWORD v24[4]; // [rsp+20h] [rbp-160h] BYREF
  char *v25; // [rsp+40h] [rbp-140h]
  char v26; // [rsp+58h] [rbp-128h] BYREF
  char *v27; // [rsp+78h] [rbp-108h]
  char v28; // [rsp+90h] [rbp-F0h] BYREF
  char *v29; // [rsp+B8h] [rbp-C8h]
  char v30; // [rsp+C8h] [rbp-B8h] BYREF
  char *v31; // [rsp+100h] [rbp-80h]
  char v32; // [rsp+110h] [rbp-70h] BYREF
  unsigned __int64 v33; // [rsp+148h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208AC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_39;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208AC);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 200;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F86530 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_37;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F86530);
  v11 = *(__int64 **)(a1 + 8);
  v12 = *(_QWORD *)(v10 + 176);
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_38:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_5027190 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_38;
  }
  v15 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(
                      *(_QWORD *)(v13 + 8),
                      &unk_5027190)
                  + 256);
  v16 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v17 = 0;
  if ( v16 != sub_2DAC790 )
    v17 = v16();
  v23[3] = v15;
  v23[0] = v17;
  v23[1] = v7;
  v23[2] = v12;
  sub_2F5FEE0(v24);
  v18 = sub_2F3AD20(v23, a2);
  if ( v33 )
    j_j___libc_free_0_0(v33);
  if ( v31 != &v32 )
    _libc_free((unsigned __int64)v31);
  if ( v29 != &v30 )
    _libc_free((unsigned __int64)v29);
  if ( v27 != &v28 )
    _libc_free((unsigned __int64)v27);
  if ( v25 != &v26 )
    _libc_free((unsigned __int64)v25);
  v19 = v24[0];
  if ( v24[0] )
  {
    v20 = v24[0] + 24LL * *(_QWORD *)(v24[0] - 8LL);
    if ( v24[0] != v20 )
    {
      do
      {
        v21 = *(_QWORD *)(v20 - 8);
        v20 -= 24;
        if ( v21 )
          j_j___libc_free_0_0(v21);
      }
      while ( v19 != v20 );
    }
    j_j_j___libc_free_0_0(v19 - 8);
  }
  return v18;
}
