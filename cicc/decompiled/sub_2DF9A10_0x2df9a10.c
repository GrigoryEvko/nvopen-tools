// Function: sub_2DF9A10
// Address: 0x2df9a10
//
__int64 __fastcall sub_2DF9A10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r8d
  __int64 v5; // r13
  _QWORD **v6; // rdx
  _QWORD **i; // rbx
  _QWORD *v8; // r12
  __int64 v9; // rax
  unsigned __int8 v10; // cl
  __int64 v11; // rcx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // r9
  unsigned __int64 v15; // r15
  unsigned __int8 *v16; // rsi
  unsigned __int64 *v17; // r12
  __int64 v18; // rdx
  unsigned __int64 *v19; // r14
  int v20; // eax
  unsigned __int64 *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int8 v25; // di
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // r13
  __int64 v30; // rsi
  unsigned __int64 *v31; // rdx
  unsigned __int64 *v32; // rdi
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rbx
  __int64 v35; // rsi
  int v36; // eax
  __int64 v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  char v40; // [rsp+18h] [rbp-58h]
  _QWORD **v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+28h] [rbp-48h]
  unsigned __int64 v44; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v45[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = 0;
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != 1 || **(_BYTE **)(a2 + 32) != 14 )
    return v3;
  v5 = sub_2E89120(a2);
  v6 = *(_QWORD ***)(a1 + 1080);
  v42 = &v6[*(unsigned int *)(a1 + 1088)];
  if ( v42 != v6 )
  {
    for ( i = *(_QWORD ***)(a1 + 1080); v42 != i; ++i )
    {
      v8 = *i;
      v9 = sub_B10CD0(a2 + 56);
      v10 = *(_BYTE *)(v9 - 16);
      if ( (v10 & 2) != 0 )
      {
        if ( *(_DWORD *)(v9 - 24) == 2 )
        {
          v23 = *(_QWORD *)(v9 - 32);
          goto LABEL_25;
        }
      }
      else if ( ((*(_WORD *)(v9 - 16) >> 6) & 0xF) == 2 )
      {
        v23 = v9 - 16 - 8LL * ((v10 >> 2) & 0xF);
LABEL_25:
        v11 = *(_QWORD *)(v23 + 8);
        if ( v5 != *v8 )
          continue;
        goto LABEL_26;
      }
      v11 = 0;
      if ( v5 != *v8 )
        continue;
LABEL_26:
      v37 = v11;
      v24 = sub_B10CD0((__int64)(v8 + 1));
      v25 = *(_BYTE *)(v24 - 16);
      if ( (v25 & 2) != 0 )
      {
        if ( *(_DWORD *)(v24 - 24) == 2 )
        {
          v27 = *(_QWORD *)(v24 - 32);
LABEL_35:
          v26 = *(_QWORD *)(v27 + 8);
          goto LABEL_29;
        }
      }
      else if ( ((*(_WORD *)(v24 - 16) >> 6) & 0xF) == 2 )
      {
        v27 = v24 - 16 - 8LL * ((v25 >> 2) & 0xF);
        goto LABEL_35;
      }
      v26 = 0;
LABEL_29:
      if ( v37 == v26 && v8[2] == a3 )
        return 1;
    }
  }
  v12 = *(_QWORD *)(a2 + 56);
  v45[0] = v12;
  if ( v12 )
    sub_B96E90((__int64)v45, v12, 1);
  v13 = (_QWORD *)sub_22077B0(0x18u);
  v15 = (unsigned __int64)v13;
  if ( v13 )
  {
    v16 = (unsigned __int8 *)v45[0];
    *v13 = v5;
    v13[1] = v16;
    if ( v16 )
      sub_B976B0((__int64)v45, v16, (__int64)(v13 + 1));
    *(_QWORD *)(v15 + 16) = a3;
    v44 = v15;
  }
  else
  {
    v44 = 0;
    if ( v45[0] )
      sub_B91220((__int64)v45, v45[0]);
  }
  v17 = &v44;
  v18 = *(unsigned int *)(a1 + 1088);
  v19 = *(unsigned __int64 **)(a1 + 1080);
  v20 = *(_DWORD *)(a1 + 1088);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1092) )
  {
    if ( v19 > &v44 || &v44 >= &v19[v18] )
    {
      v43 = -1;
      v40 = 0;
    }
    else
    {
      v40 = 1;
      v43 = &v44 - v19;
    }
    v28 = (unsigned __int64 *)sub_C8D7D0(a1 + 1080, a1 + 1096, v18 + 1, 8u, v45, v14);
    v29 = *(_QWORD *)(a1 + 1080);
    v19 = v28;
    v30 = *(unsigned int *)(a1 + 1088);
    v31 = (unsigned __int64 *)v29;
    v32 = &v28[v30];
    if ( v30 * 8 )
    {
      do
      {
        if ( v28 )
        {
          *v28 = *v31;
          *v31 = 0;
        }
        ++v28;
        ++v31;
      }
      while ( v28 != v32 );
      v29 = *(_QWORD *)(a1 + 1080);
      v33 = v29 + 8LL * *(unsigned int *)(a1 + 1088);
      if ( v29 != v33 )
      {
        do
        {
          v34 = *(_QWORD *)(v33 - 8);
          v33 -= 8LL;
          if ( v34 )
          {
            v35 = *(_QWORD *)(v34 + 8);
            if ( v35 )
              sub_B91220(v34 + 8, v35);
            j_j___libc_free_0(v34);
          }
        }
        while ( v29 != v33 );
        v29 = *(_QWORD *)(a1 + 1080);
      }
    }
    v36 = v45[0];
    if ( a1 + 1096 != v29 )
    {
      v38 = v45[0];
      _libc_free(v29);
      v36 = v38;
    }
    v17 = &v44;
    *(_DWORD *)(a1 + 1092) = v36;
    *(_QWORD *)(a1 + 1080) = v19;
    v18 = *(unsigned int *)(a1 + 1088);
    if ( v40 )
      v17 = &v19[v43];
    v20 = *(_DWORD *)(a1 + 1088);
  }
  v21 = &v19[v18];
  if ( v21 )
  {
    *v21 = *v17;
    *v17 = 0;
    v15 = v44;
    v20 = *(_DWORD *)(a1 + 1088);
  }
  *(_DWORD *)(a1 + 1088) = v20 + 1;
  if ( v15 )
  {
    v22 = *(_QWORD *)(v15 + 8);
    if ( v22 )
      sub_B91220(v15 + 8, v22);
    j_j___libc_free_0(v15);
  }
  return 1;
}
