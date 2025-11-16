// Function: sub_3721510
// Address: 0x3721510
//
__int64 __fastcall sub_3721510(__int64 a1, __int64 a2, __m128i a3)
{
  char v3; // r8
  __int64 result; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // rbx
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdi
  _QWORD *v26; // rbx
  unsigned __int64 v27; // rdi
  _QWORD v28[4]; // [rsp+0h] [rbp-A0h] BYREF
  char v29; // [rsp+20h] [rbp-80h]
  _QWORD *v30; // [rsp+28h] [rbp-78h]
  _QWORD v31[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v32; // [rsp+40h] [rbp-60h]
  __int64 v33; // [rsp+48h] [rbp-58h]
  unsigned int v34; // [rsp+50h] [rbp-50h]
  __int64 v35; // [rsp+58h] [rbp-48h]
  __int64 v36; // [rsp+60h] [rbp-40h]
  __int64 v37; // [rsp+68h] [rbp-38h]
  unsigned int v38; // [rsp+70h] [rbp-30h]
  int v39; // [rsp+78h] [rbp-28h]

  v3 = sub_BB98D0((_QWORD *)a1, a2);
  result = 0;
  if ( !v3 )
  {
    v5 = *(__int64 **)(a1 + 8);
    v6 = *v5;
    v7 = v5[1];
    if ( v6 == v7 )
      goto LABEL_21;
    while ( *(_UNKNOWN **)v6 != &unk_4F8144C )
    {
      v6 += 16;
      if ( v7 == v6 )
        goto LABEL_21;
    }
    v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F8144C);
    v9 = *(__int64 **)(a1 + 8);
    v10 = v8 + 176;
    v11 = *v9;
    v12 = v9[1];
    if ( v11 == v12 )
      goto LABEL_21;
    while ( *(_UNKNOWN **)v11 != &unk_4F875EC )
    {
      v11 += 16;
      if ( v12 == v11 )
        goto LABEL_21;
    }
    v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
            *(_QWORD *)(v11 + 8),
            &unk_4F875EC);
    v14 = *(__int64 **)(a1 + 8);
    v15 = v13 + 176;
    v16 = *v14;
    v17 = v14[1];
    if ( v16 == v17 )
LABEL_21:
      BUG();
    while ( *(_UNKNOWN **)v16 != &unk_5010CD4 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_21;
    }
    v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
            *(_QWORD *)(v16 + 8),
            &unk_5010CD4);
    v28[0] = a2;
    v19 = *(_QWORD *)(v18 + 176);
    LOBYTE(v18) = *(_BYTE *)(a1 + 169);
    v28[1] = v10;
    v28[3] = v15;
    v29 = v18;
    v28[2] = v19;
    v30 = 0;
    v31[0] = 0;
    v31[1] = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    sub_3720740(v28, (__int64)&unk_5010CD4, a3, v19, v20, v21, v22);
    sub_371E590(v31, a1 + 176);
    sub_C7D6A0(v36, 16LL * v38, 8);
    sub_C7D6A0(v32, 16LL * v34, 8);
    v23 = (_QWORD *)v31[0];
    if ( v31[0] )
    {
      do
      {
        v24 = (unsigned __int64)v23;
        v23 = (_QWORD *)*v23;
        v25 = *(_QWORD *)(v24 + 8);
        if ( v25 )
          j_j___libc_free_0(v25);
        j_j___libc_free_0(v24);
      }
      while ( v23 );
    }
    v26 = v30;
    while ( v26 )
    {
      v27 = (unsigned __int64)v26;
      v26 = (_QWORD *)*v26;
      j_j___libc_free_0(v27);
    }
    return 1;
  }
  return result;
}
