// Function: sub_2D1F640
// Address: 0x2d1f640
//
__int64 __fastcall sub_2D1F640(__int64 a1, __int64 a2)
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
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdi
  unsigned int v14; // r12d
  __int64 v15; // rsi
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  _QWORD v21[3]; // [rsp+0h] [rbp-B0h] BYREF
  int v22; // [rsp+18h] [rbp-98h] BYREF
  unsigned __int64 v23; // [rsp+20h] [rbp-90h]
  int *v24; // [rsp+28h] [rbp-88h]
  int *v25; // [rsp+30h] [rbp-80h]
  __int64 v26; // [rsp+38h] [rbp-78h]
  int v27; // [rsp+48h] [rbp-68h] BYREF
  unsigned __int64 v28; // [rsp+50h] [rbp-60h]
  int *v29; // [rsp+58h] [rbp-58h]
  int *v30; // [rsp+60h] [rbp-50h]
  __int64 v31; // [rsp+68h] [rbp-48h]
  __int64 v32; // [rsp+70h] [rbp-40h]
  __int64 v33; // [rsp+78h] [rbp-38h]
  __int64 v34; // [rsp+80h] [rbp-30h]
  __int64 v35; // [rsp+88h] [rbp-28h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_22:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_22;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 176;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_21:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F86530 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_21;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F86530);
  v21[0] = v7;
  v11 = *(_QWORD *)(v10 + 176);
  v26 = 0;
  v22 = 0;
  v21[1] = v11;
  v24 = &v22;
  v25 = &v22;
  v29 = &v27;
  v30 = &v27;
  v23 = 0;
  v27 = 0;
  v28 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  if ( LODWORD(qword_5016428[8]) && SLODWORD(qword_5016348[8]) > 0 )
  {
    v12 = sub_2D1E1A0(v21, a2);
    v13 = v33;
    v14 = v12;
    v15 = 16LL * (unsigned int)v35;
  }
  else
  {
    v13 = 0;
    v15 = 0;
    v14 = 0;
  }
  sub_C7D6A0(v13, v15, 8);
  v16 = v28;
  while ( v16 )
  {
    sub_2D1B830(*(_QWORD *)(v16 + 24));
    v17 = v16;
    v16 = *(_QWORD *)(v16 + 16);
    j_j___libc_free_0(v17);
  }
  v18 = v23;
  while ( v18 )
  {
    sub_2D1B830(*(_QWORD *)(v18 + 24));
    v19 = v18;
    v18 = *(_QWORD *)(v18 + 16);
    j_j___libc_free_0(v19);
  }
  return v14;
}
