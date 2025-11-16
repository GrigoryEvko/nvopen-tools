// Function: sub_1D596C0
// Address: 0x1d596c0
//
void __fastcall sub_1D596C0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  _WORD *v6; // rdx
  size_t v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rdi
  _DWORD *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rdi
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  _WORD *v19; // rdx
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // [rsp-A8h] [rbp-A8h]
  char *v27; // [rsp-98h] [rbp-98h] BYREF
  size_t v28; // [rsp-90h] [rbp-90h]
  _QWORD v29[2]; // [rsp-88h] [rbp-88h] BYREF
  __int64 v30[2]; // [rsp-78h] [rbp-78h] BYREF
  _QWORD v31[2]; // [rsp-68h] [rbp-68h] BYREF
  char *v32; // [rsp-58h] [rbp-58h] BYREF
  size_t v33; // [rsp-50h] [rbp-50h]
  _QWORD v34[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( !*(_QWORD *)(a1 + 624) )
    return;
  v30[0] = (__int64)v31;
  sub_1D59170(v30, "GraphRoot", (__int64)"");
  v27 = (char *)v29;
  sub_1D59170((__int64 *)&v27, "plaintext=circle", (__int64)"");
  v3 = *a2;
  v4 = *(_QWORD *)(*a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a2 + 16) - v4) <= 4 )
  {
    v3 = sub_16E7EE0(v3, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v4 = 1685016073;
    *(_BYTE *)(v4 + 4) = 101;
    *(_QWORD *)(v3 + 24) += 5LL;
  }
  v5 = sub_16E7B40(v3, 0);
  v6 = *(_WORD **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 1u )
  {
    sub_16E7EE0(v5, (char *)"[ ", 2u);
    v7 = v28;
    if ( !v28 )
      goto LABEL_6;
  }
  else
  {
    *v6 = 8283;
    v7 = v28;
    *(_QWORD *)(v5 + 24) += 2LL;
    if ( !v7 )
      goto LABEL_6;
  }
  v20 = sub_16E7EE0(*a2, v27, v7);
  v21 = *(_BYTE **)(v20 + 24);
  if ( *(_BYTE **)(v20 + 16) == v21 )
  {
    sub_16E7EE0(v20, ",", 1u);
  }
  else
  {
    *v21 = 44;
    ++*(_QWORD *)(v20 + 24);
  }
LABEL_6:
  v8 = *a2;
  v9 = *(_QWORD *)(*a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a2 + 16) - v9) <= 8 )
  {
    sub_16E7EE0(v8, " label =\"", 9u);
  }
  else
  {
    *(_BYTE *)(v9 + 8) = 34;
    *(_QWORD *)v9 = 0x3D206C6562616C20LL;
    *(_QWORD *)(v8 + 24) += 9LL;
  }
  v26 = *a2;
  sub_16BE9B0((__int64 *)&v32, (__int64)v30);
  sub_16E7EE0(v26, v32, v33);
  if ( v32 != (char *)v34 )
    j_j___libc_free_0(v32, v34[0] + 1LL);
  v10 = *a2;
  v11 = *(_DWORD **)(*a2 + 24);
  if ( *(_QWORD *)(*a2 + 16) - (_QWORD)v11 <= 3u )
  {
    sub_16E7EE0(v10, "\"];\n", 4u);
  }
  else
  {
    *v11 = 171662626;
    *(_QWORD *)(v10 + 24) += 4LL;
  }
  if ( v27 != (char *)v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  if ( (_QWORD *)v30[0] != v31 )
    j_j___libc_free_0(v30[0], v31[0] + 1LL);
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 624) + 176LL);
  if ( v12 && *(_DWORD *)(v12 + 28) != -1 )
  {
    v32 = (char *)v34;
    sub_1D59170((__int64 *)&v32, "color=blue,style=dashed", (__int64)"");
    v13 = *a2;
    v14 = *(_QWORD *)(*a2 + 24);
    v15 = *(_QWORD *)(a1 + 48) + 272LL * *(int *)(v12 + 28);
    if ( (unsigned __int64)(*(_QWORD *)(*a2 + 16) - v14) <= 4 )
    {
      v13 = sub_16E7EE0(v13, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v14 = 1685016073;
      *(_BYTE *)(v14 + 4) = 101;
      *(_QWORD *)(v13 + 24) += 5LL;
    }
    sub_16E7B40(v13, 0);
    v16 = *a2;
    v17 = *(_QWORD **)(*a2 + 24);
    if ( *(_QWORD *)(*a2 + 16) - (_QWORD)v17 <= 7u )
    {
      v16 = sub_16E7EE0(v16, " -> Node", 8u);
    }
    else
    {
      *v17 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v16 + 24) += 8LL;
    }
    sub_16E7B40(v16, v15);
    if ( v33 )
    {
      v22 = *a2;
      v23 = *(_BYTE **)(*a2 + 24);
      if ( *(_BYTE **)(*a2 + 16) == v23 )
      {
        v22 = sub_16E7EE0(v22, "[", 1u);
      }
      else
      {
        *v23 = 91;
        ++*(_QWORD *)(v22 + 24);
      }
      v24 = sub_16E7EE0(v22, v32, v33);
      v25 = *(_BYTE **)(v24 + 24);
      if ( *(_BYTE **)(v24 + 16) == v25 )
      {
        sub_16E7EE0(v24, "]", 1u);
      }
      else
      {
        *v25 = 93;
        ++*(_QWORD *)(v24 + 24);
      }
    }
    v18 = *a2;
    v19 = *(_WORD **)(*a2 + 24);
    if ( *(_QWORD *)(*a2 + 16) - (_QWORD)v19 <= 1u )
    {
      sub_16E7EE0(v18, ";\n", 2u);
    }
    else
    {
      *v19 = 2619;
      *(_QWORD *)(v18 + 24) += 2LL;
    }
    if ( v32 != (char *)v34 )
      j_j___libc_free_0(v32, v34[0] + 1LL);
  }
}
