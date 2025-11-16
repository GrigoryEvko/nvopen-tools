// Function: sub_1CB8AE0
// Address: 0x1cb8ae0
//
__int64 __fastcall sub_1CB8AE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // r13d
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v12; // [rsp+0h] [rbp-150h] BYREF
  __int64 v13; // [rsp+10h] [rbp-140h] BYREF
  __int64 v14; // [rsp+18h] [rbp-138h]
  __int64 *v15; // [rsp+20h] [rbp-130h]
  __int64 *v16; // [rsp+28h] [rbp-128h]
  __int64 v17; // [rsp+30h] [rbp-120h]
  int v18; // [rsp+40h] [rbp-110h] BYREF
  __int64 v19; // [rsp+48h] [rbp-108h]
  int *v20; // [rsp+50h] [rbp-100h]
  int *v21; // [rsp+58h] [rbp-F8h]
  __int64 v22; // [rsp+60h] [rbp-F0h]
  int v23; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v24; // [rsp+78h] [rbp-D8h]
  int *v25; // [rsp+80h] [rbp-D0h]
  int *v26; // [rsp+88h] [rbp-C8h]
  __int64 v27; // [rsp+90h] [rbp-C0h]
  int v28; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+A8h] [rbp-A8h]
  int *v30; // [rsp+B0h] [rbp-A0h]
  int *v31; // [rsp+B8h] [rbp-98h]
  __int64 v32; // [rsp+C0h] [rbp-90h]
  int v33; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v34; // [rsp+D8h] [rbp-78h]
  int *v35; // [rsp+E0h] [rbp-70h]
  int *v36; // [rsp+E8h] [rbp-68h]
  __int64 v37; // [rsp+F0h] [rbp-60h]
  __int64 v38; // [rsp+F8h] [rbp-58h]
  __int64 v39; // [rsp+100h] [rbp-50h]
  __int64 v40; // [rsp+108h] [rbp-48h]
  unsigned int v41; // [rsp+110h] [rbp-40h]
  __int64 v42; // [rsp+118h] [rbp-38h]
  __int64 v43; // [rsp+120h] [rbp-30h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_15:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_15;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  v15 = &v13;
  v16 = &v13;
  v20 = &v18;
  v21 = &v18;
  v25 = &v23;
  v26 = &v23;
  v30 = &v28;
  v31 = &v28;
  v35 = &v33;
  v36 = &v33;
  v12 = 0x100000000LL;
  v13 = 0;
  v14 = 0;
  v17 = 0;
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
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v6 = sub_1CB7E90((__int64)&v12, a2, v5 + 160);
  if ( v41 )
  {
    v7 = v39;
    v8 = v39 + 560LL * v41;
    do
    {
      if ( *(_QWORD *)v7 != -16 && *(_QWORD *)v7 != -8 && (*(_BYTE *)(v7 + 16) & 1) == 0 )
        j___libc_free_0(*(_QWORD *)(v7 + 24));
      v7 += 560;
    }
    while ( v8 != v7 );
  }
  j___libc_free_0(v39);
  v9 = v34;
  while ( v9 )
  {
    sub_1CB6920(*(_QWORD *)(v9 + 24));
    v10 = v9;
    v9 = *(_QWORD *)(v9 + 16);
    j_j___libc_free_0(v10, 48);
  }
  sub_1CB6AF0(v29);
  sub_1CB6CC0(v24);
  sub_1CB6CC0(v19);
  sub_1CB6AF0(v14);
  return v6;
}
