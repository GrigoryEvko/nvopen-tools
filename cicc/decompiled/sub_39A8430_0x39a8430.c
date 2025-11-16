// Function: sub_39A8430
// Address: 0x39a8430
//
__int64 __fastcall sub_39A8430(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v4; // r14
  __int64 v5; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // r14
  void *v17; // rcx
  size_t v18; // rdx
  size_t v19; // r8
  void *v20; // rcx
  size_t v21; // rdx
  size_t v22; // r8
  void *v23; // rcx
  size_t v24; // rdx
  size_t v25; // r8
  void *v26; // rcx
  size_t v27; // rdx
  size_t v28; // r8
  __int64 v29; // rdx
  void (__fastcall *v30)(__int64 *, __int64, __int64, __int64, __int64); // r14
  __int64 v31; // r15
  __int64 v32; // rsi
  __int64 v33; // rdx

  v4 = sub_39A81B0((__int64)a1, *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
  v5 = (__int64)sub_39A23D0((__int64)a1, (unsigned __int8 *)a2);
  if ( !v5 )
  {
    v5 = sub_39A5A90((__int64)a1, 30, (__int64)v4, (unsigned __int8 *)a2);
    v7 = *(unsigned int *)(a2 + 8);
    v8 = *(_QWORD *)(a2 + 8 * (1 - v7));
    if ( v8 )
    {
      sub_161E970(v8);
      v7 = *(unsigned int *)(a2 + 8);
      if ( v9 )
      {
        v26 = *(void **)(a2 + 8 * (1 - v7));
        if ( v26 )
        {
          v26 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v7)));
          v28 = v27;
        }
        else
        {
          v28 = 0;
        }
        sub_39A3F30(a1, v5, 3, v26, v28);
        v29 = *(unsigned int *)(a2 + 8);
        v30 = *(void (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*a1 + 8);
        v31 = *(_QWORD *)(a2 - 8 * v29);
        v32 = *(_QWORD *)(a2 + 8 * (1 - v29));
        if ( v32 )
          v32 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v29)));
        else
          v33 = 0;
        v30(a1, v32, v33, v5, v31);
        v7 = *(unsigned int *)(a2 + 8);
      }
    }
    v10 = *(_QWORD *)(a2 + 8 * (2 - v7));
    if ( v10 )
    {
      sub_161E970(v10);
      v7 = *(unsigned int *)(a2 + 8);
      if ( v11 )
      {
        v23 = *(void **)(a2 + 8 * (2 - v7));
        if ( v23 )
        {
          v23 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v7)));
          v25 = v24;
        }
        else
        {
          v25 = 0;
        }
        sub_39A3F30(a1, v5, 15873, v23, v25);
        v7 = *(unsigned int *)(a2 + 8);
      }
    }
    v12 = *(_QWORD *)(a2 + 8 * (3 - v7));
    if ( v12 )
    {
      sub_161E970(v12);
      v7 = *(unsigned int *)(a2 + 8);
      if ( v13 )
      {
        v20 = *(void **)(a2 + 8 * (3 - v7));
        if ( v20 )
        {
          v20 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v7)));
          v22 = v21;
        }
        else
        {
          v22 = 0;
        }
        sub_39A3F30(a1, v5, 15872, v20, v22);
        v7 = *(unsigned int *)(a2 + 8);
      }
    }
    v14 = *(_QWORD *)(a2 + 8 * (4 - v7));
    if ( v14 )
    {
      sub_161E970(v14);
      if ( v15 )
      {
        v16 = 4LL - *(unsigned int *)(a2 + 8);
        v17 = *(void **)(a2 + 8 * v16);
        if ( v17 )
        {
          v17 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * v16));
          v19 = v18;
        }
        else
        {
          v19 = 0;
        }
        sub_39A3F30(a1, v5, 15874, v17, v19);
      }
    }
  }
  return v5;
}
