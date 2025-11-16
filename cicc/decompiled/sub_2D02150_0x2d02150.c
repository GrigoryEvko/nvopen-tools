// Function: sub_2D02150
// Address: 0x2d02150
//
__int64 __fastcall sub_2D02150(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // eax
  unsigned __int64 v7; // rbx
  unsigned int v8; // r12d
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  unsigned __int64 v17; // [rsp+0h] [rbp-120h] BYREF
  __int64 v18; // [rsp+10h] [rbp-110h] BYREF
  unsigned __int64 v19; // [rsp+18h] [rbp-108h]
  __int64 *v20; // [rsp+20h] [rbp-100h]
  __int64 *v21; // [rsp+28h] [rbp-F8h]
  __int64 v22; // [rsp+30h] [rbp-F0h]
  int v23; // [rsp+40h] [rbp-E0h] BYREF
  unsigned __int64 v24; // [rsp+48h] [rbp-D8h]
  int *v25; // [rsp+50h] [rbp-D0h]
  int *v26; // [rsp+58h] [rbp-C8h]
  __int64 v27; // [rsp+60h] [rbp-C0h]
  int v28; // [rsp+70h] [rbp-B0h] BYREF
  unsigned __int64 v29; // [rsp+78h] [rbp-A8h]
  int *v30; // [rsp+80h] [rbp-A0h]
  int *v31; // [rsp+88h] [rbp-98h]
  __int64 v32; // [rsp+90h] [rbp-90h]
  int v33; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v34; // [rsp+A8h] [rbp-78h]
  int *v35; // [rsp+B0h] [rbp-70h]
  int *v36; // [rsp+B8h] [rbp-68h]
  __int64 v37; // [rsp+C0h] [rbp-60h]
  int v38; // [rsp+D0h] [rbp-50h] BYREF
  unsigned __int64 v39; // [rsp+D8h] [rbp-48h]
  int *v40; // [rsp+E0h] [rbp-40h]
  int *v41; // [rsp+E8h] [rbp-38h]
  __int64 v42; // [rsp+F0h] [rbp-30h]
  __int64 v43; // [rsp+F8h] [rbp-28h]
  __int64 v44; // [rsp+100h] [rbp-20h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_14:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_14;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v20 = &v18;
  v21 = &v18;
  v25 = &v23;
  v26 = &v23;
  v30 = &v28;
  v31 = &v28;
  v35 = &v33;
  v36 = &v33;
  v40 = &v38;
  v41 = &v38;
  v17 = 0x100000000LL;
  v18 = 0;
  v19 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v6 = sub_2D01410(&v17, a2, v5 + 176);
  v7 = v39;
  v8 = v6;
  while ( v7 )
  {
    sub_2CFFFC0(*(_QWORD *)(v7 + 24));
    v9 = v7;
    v7 = *(_QWORD *)(v7 + 16);
    j_j___libc_free_0(v9);
  }
  v10 = v34;
  while ( v10 )
  {
    sub_2D00360(*(_QWORD *)(v10 + 24));
    v11 = v10;
    v10 = *(_QWORD *)(v10 + 16);
    j_j___libc_free_0(v11);
  }
  v12 = v29;
  while ( v12 )
  {
    sub_2D00190(*(_QWORD *)(v12 + 24));
    v13 = v12;
    v12 = *(_QWORD *)(v12 + 16);
    j_j___libc_free_0(v13);
  }
  v14 = v24;
  while ( v14 )
  {
    sub_2D00190(*(_QWORD *)(v14 + 24));
    v15 = v14;
    v14 = *(_QWORD *)(v14 + 16);
    j_j___libc_free_0(v15);
  }
  sub_2D00360(v19);
  return v8;
}
