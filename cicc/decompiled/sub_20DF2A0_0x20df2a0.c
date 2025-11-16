// Function: sub_20DF2A0
// Address: 0x20df2a0
//
__int64 __fastcall sub_20DF2A0(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  char v5; // r13
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r13
  __int64 *v22; // rdi
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 (*v25)(void); // rdx
  __int64 (*v26)(); // rax
  unsigned int v27; // r12d
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-170h] BYREF
  __int64 v31; // [rsp+8h] [rbp-168h]
  __int64 v32; // [rsp+10h] [rbp-160h]
  __int64 v33; // [rsp+18h] [rbp-158h]
  int v34; // [rsp+20h] [rbp-150h]
  _QWORD v35[5]; // [rsp+30h] [rbp-140h] BYREF
  unsigned __int64 v36; // [rsp+58h] [rbp-118h]
  __int64 v37; // [rsp+88h] [rbp-E8h]
  __int64 v38; // [rsp+A0h] [rbp-D0h]
  __int64 v39; // [rsp+B0h] [rbp-C0h]
  char *v40; // [rsp+F0h] [rbp-80h]
  char v41; // [rsp+100h] [rbp-70h] BYREF
  unsigned __int64 v42; // [rsp+120h] [rbp-50h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_38:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FCBA30 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_38;
  }
  v5 = 0;
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FCBA30);
  if ( (*(_BYTE *)(a2[1] + 640LL) & 1) == 0 )
    v5 = *(_BYTE *)(v6 + 226);
  v7 = *(__int64 **)(a1 + 8);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4FC453D )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_36;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4FC453D);
  v11 = *(__int64 **)(a1 + 8);
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v30 = v10;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4FC5828 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_37;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4FC5828);
  sub_20D6CE0((__int64)v35, v5, 1, (__int64)&v30, v14, 0);
  v16 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC6A0E, v15);
  v21 = v16;
  if ( v16 )
    v21 = (*(__int64 (__fastcall **)(__int64, void *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, int))(*(_QWORD *)v16 + 104LL))(
            v16,
            &unk_4FC6A0E,
            v17,
            v18,
            v19,
            v20,
            v30,
            v31,
            v32,
            v33,
            v34);
  v22 = (__int64 *)a2[2];
  v23 = 0;
  v24 = *v22;
  v25 = *(__int64 (**)(void))(*v22 + 112);
  if ( v25 != sub_1D00B10 )
  {
    v23 = v25();
    v24 = *(_QWORD *)a2[2];
  }
  v26 = *(__int64 (**)())(v24 + 40);
  if ( v26 == sub_1D00B00 || (v29 = v26()) == 0 )
    v27 = 0;
  else
    v27 = sub_20DEC90((__int64)v35, a2, v29, v23, v21, 0, 0);
  _libc_free(v42);
  if ( v40 != &v41 )
    _libc_free((unsigned __int64)v40);
  if ( v38 )
    j_j___libc_free_0(v38, v39 - v38);
  j___libc_free_0(v37);
  if ( v36 != v35[4] )
    _libc_free(v36);
  if ( v35[0] )
    j_j___libc_free_0(v35[0], v35[2] - v35[0]);
  j___libc_free_0(v32);
  return v27;
}
