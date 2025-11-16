// Function: sub_34330D0
// Address: 0x34330d0
//
void __fastcall sub_34330D0(__int64 a1, __int64 *a2)
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
  __int64 v12; // r14
  __int64 v13; // rax
  __m128i si128; // xmm0
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned __int64 v17; // r12
  __int64 v18; // rdi
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  _WORD *v21; // rdx
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rdi
  _BYTE *v27; // rax
  __int64 v28; // [rsp-A8h] [rbp-A8h]
  __int64 v29; // [rsp-98h] [rbp-98h] BYREF
  unsigned __int64 v30; // [rsp-90h] [rbp-90h]
  _QWORD v31[2]; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v32[2]; // [rsp-78h] [rbp-78h] BYREF
  char v33[16]; // [rsp-68h] [rbp-68h] BYREF
  unsigned __int64 v34; // [rsp-58h] [rbp-58h] BYREF
  size_t v35; // [rsp-50h] [rbp-50h]
  _QWORD v36[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( !*(_QWORD *)(a1 + 592) )
    return;
  v32[0] = (unsigned __int64)v33;
  strcpy(v33, "GraphRoot");
  v32[1] = 9;
  v29 = (__int64)v31;
  v34 = 16;
  v29 = sub_22409D0((__int64)&v29, &v34, 0);
  v31[0] = v34;
  *(__m128i *)v29 = _mm_load_si128((const __m128i *)&xmmword_44E1320);
  v30 = v34;
  *(_BYTE *)(v29 + v34) = 0;
  v3 = *a2;
  v4 = *(_QWORD *)(*a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v4) <= 4 )
  {
    v3 = sub_CB6200(v3, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v4 = 1685016073;
    *(_BYTE *)(v4 + 4) = 101;
    *(_QWORD *)(v3 + 32) += 5LL;
  }
  v5 = sub_CB5A80(v3, 0);
  v6 = *(_WORD **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 1u )
  {
    sub_CB6200(v5, (unsigned __int8 *)"[ ", 2u);
    v7 = v30;
    if ( !v30 )
      goto LABEL_6;
  }
  else
  {
    *v6 = 8283;
    v7 = v30;
    *(_QWORD *)(v5 + 32) += 2LL;
    if ( !v7 )
      goto LABEL_6;
  }
  v22 = sub_CB6200(*a2, (unsigned __int8 *)v29, v7);
  v23 = *(_BYTE **)(v22 + 32);
  if ( *(_BYTE **)(v22 + 24) == v23 )
  {
    sub_CB6200(v22, (unsigned __int8 *)",", 1u);
  }
  else
  {
    *v23 = 44;
    ++*(_QWORD *)(v22 + 32);
  }
LABEL_6:
  v8 = *a2;
  v9 = *(_QWORD *)(*a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v9) <= 8 )
  {
    sub_CB6200(v8, " label =\"", 9u);
  }
  else
  {
    *(_BYTE *)(v9 + 8) = 34;
    *(_QWORD *)v9 = 0x3D206C6562616C20LL;
    *(_QWORD *)(v8 + 32) += 9LL;
  }
  v28 = *a2;
  sub_C67200((__int64 *)&v34, (__int64)v32);
  sub_CB6200(v28, (unsigned __int8 *)v34, v35);
  if ( (_QWORD *)v34 != v36 )
    j_j___libc_free_0(v34);
  v10 = *a2;
  v11 = *(_DWORD **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v11 <= 3u )
  {
    sub_CB6200(v10, (unsigned __int8 *)"\"];\n", 4u);
  }
  else
  {
    *v11 = 171662626;
    *(_QWORD *)(v10 + 32) += 4LL;
  }
  if ( (_QWORD *)v29 != v31 )
    j_j___libc_free_0(v29);
  if ( (char *)v32[0] != v33 )
    j_j___libc_free_0(v32[0]);
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 592) + 384LL);
  if ( v12 && *(_DWORD *)(v12 + 36) != -1 )
  {
    v32[0] = 23;
    v34 = (unsigned __int64)v36;
    v13 = sub_22409D0((__int64)&v34, v32, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_44E1330);
    v34 = v13;
    v36[0] = v32[0];
    *(_DWORD *)(v13 + 16) = 1935762493;
    *(_WORD *)(v13 + 20) = 25960;
    *(_BYTE *)(v13 + 22) = 100;
    *(__m128i *)v13 = si128;
    v35 = v32[0];
    *(_BYTE *)(v34 + v32[0]) = 0;
    v15 = *a2;
    v16 = *(_QWORD *)(*a2 + 32);
    v17 = *(_QWORD *)(a1 + 48) + ((__int64)*(int *)(v12 + 36) << 8);
    if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v16) <= 4 )
    {
      v15 = sub_CB6200(v15, "\tNode", 5u);
    }
    else
    {
      *(_DWORD *)v16 = 1685016073;
      *(_BYTE *)(v16 + 4) = 101;
      *(_QWORD *)(v15 + 32) += 5LL;
    }
    sub_CB5A80(v15, 0);
    v18 = *a2;
    v19 = *(_QWORD **)(*a2 + 32);
    if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v19 <= 7u )
    {
      v18 = sub_CB6200(v18, " -> Node", 8u);
    }
    else
    {
      *v19 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v18 + 32) += 8LL;
    }
    sub_CB5A80(v18, v17);
    if ( v35 )
    {
      v24 = *a2;
      v25 = *(_BYTE **)(*a2 + 32);
      if ( *(_BYTE **)(*a2 + 24) == v25 )
      {
        v24 = sub_CB6200(v24, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v25 = 91;
        ++*(_QWORD *)(v24 + 32);
      }
      v26 = sub_CB6200(v24, (unsigned __int8 *)v34, v35);
      v27 = *(_BYTE **)(v26 + 32);
      if ( *(_BYTE **)(v26 + 24) == v27 )
      {
        sub_CB6200(v26, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v27 = 93;
        ++*(_QWORD *)(v26 + 32);
      }
    }
    v20 = *a2;
    v21 = *(_WORD **)(*a2 + 32);
    if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v21 <= 1u )
    {
      sub_CB6200(v20, (unsigned __int8 *)";\n", 2u);
    }
    else
    {
      *v21 = 2619;
      *(_QWORD *)(v20 + 32) += 2LL;
    }
    if ( (_QWORD *)v34 != v36 )
      j_j___libc_free_0(v34);
  }
}
