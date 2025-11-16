// Function: sub_315C7D0
// Address: 0x315c7d0
//
void __fastcall sub_315C7D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v8; // rax
  unsigned __int64 v9; // r13
  __int64 *v10; // r15
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // rsi
  _BYTE *v16; // r8
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdi
  _QWORD *v23; // rdx
  __int64 v24; // rdi
  _WORD *v25; // rdx
  int v26; // edi
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rsi
  int v30; // edi
  unsigned int v31; // eax
  __int64 v32; // rsi
  _BYTE *v33; // r8
  _QWORD *v34; // rsi
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rdi
  _BYTE *v38; // rax
  _BOOL8 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int8 *v42; // [rsp+30h] [rbp-80h] BYREF
  size_t v43; // [rsp+38h] [rbp-78h]
  _DWORD v44[4]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v45; // [rsp+50h] [rbp-60h]
  int v46; // [rsp+58h] [rbp-58h]
  _BYTE v47[80]; // [rsp+60h] [rbp-50h] BYREF

  v8 = sub_B46EC0(a4, a5);
  if ( !v8 )
    return;
  v9 = v8;
  v10 = *(__int64 **)a1[1];
  v41 = sub_B46EC0(a4, a5);
  v11 = v41;
  sub_315C600((__int64)&v42, *v10, a2, v12, v13, v14);
  if ( v44[0] )
  {
    if ( v44[2] )
    {
      v26 = 1;
      v40 = 1;
      v27 = (v44[2] - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v28 = *(_QWORD *)(v43 + 8LL * v27);
      if ( v41 == v28 )
      {
LABEL_20:
        v16 = v45;
        goto LABEL_4;
      }
      while ( v28 != -4096 )
      {
        v27 = (v44[2] - 1) & (v26 + v27);
        v28 = *(_QWORD *)(v43 + 8LL * v27);
        if ( v41 == v28 )
        {
          v40 = 1;
          goto LABEL_20;
        }
        ++v26;
      }
    }
    v40 = 0;
    goto LABEL_20;
  }
  v15 = &v45[v46];
  v40 = v15 != sub_3157E90(v45, (__int64)v15, &v41);
LABEL_4:
  if ( v16 != v47 )
    _libc_free((unsigned __int64)v16);
  sub_C7D6A0(v43, 8LL * v44[2], 8);
  if ( v40 )
  {
    v42 = (unsigned __int8 *)v44;
    strcpy((char *)v44, "color=red");
    v43 = 9;
  }
  else
  {
    v29 = *v10;
    v41 = a2;
    sub_315C600((__int64)&v42, v29, v11, v17, v18, v19);
    if ( v44[0] )
    {
      if ( v44[2] )
      {
        v30 = 1;
        v31 = (v44[2] - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v32 = *(_QWORD *)(v43 + 8LL * v31);
        if ( a2 == v32 )
        {
LABEL_24:
          v40 = 1;
        }
        else
        {
          while ( v32 != -4096 )
          {
            v31 = (v44[2] - 1) & (v30 + v31);
            v32 = *(_QWORD *)(v43 + 8LL * v31);
            if ( a2 == v32 )
              goto LABEL_24;
            ++v30;
          }
        }
      }
      v33 = v45;
    }
    else
    {
      v34 = &v45[v46];
      v40 = v34 != sub_3157E90(v45, (__int64)v34, &v41);
    }
    if ( v33 != v47 )
      _libc_free((unsigned __int64)v33);
    sub_C7D6A0(v43, 8LL * v44[2], 8);
    v42 = (unsigned __int8 *)v44;
    if ( v40 )
    {
      v43 = 10;
      strcpy((char *)v44, "color=blue");
    }
    else
    {
      v43 = 0;
      LOBYTE(v44[0]) = 0;
    }
  }
  v20 = *a1;
  v21 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v21) <= 4 )
  {
    v20 = sub_CB6200(v20, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v21 = 1685016073;
    *(_BYTE *)(v21 + 4) = 101;
    *(_QWORD *)(v20 + 32) += 5LL;
  }
  sub_CB5A80(v20, a2);
  v22 = *a1;
  v23 = *(_QWORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v23 <= 7u )
  {
    v22 = sub_CB6200(v22, " -> Node", 8u);
  }
  else
  {
    *v23 = 0x65646F4E203E2D20LL;
    *(_QWORD *)(v22 + 32) += 8LL;
  }
  sub_CB5A80(v22, v9);
  if ( v43 )
  {
    v35 = *a1;
    v36 = *(_BYTE **)(*a1 + 32);
    if ( *(_BYTE **)(*a1 + 24) == v36 )
    {
      v35 = sub_CB6200(v35, (unsigned __int8 *)"[", 1u);
    }
    else
    {
      *v36 = 91;
      ++*(_QWORD *)(v35 + 32);
    }
    v37 = sub_CB6200(v35, v42, v43);
    v38 = *(_BYTE **)(v37 + 32);
    if ( *(_BYTE **)(v37 + 24) == v38 )
    {
      sub_CB6200(v37, (unsigned __int8 *)"]", 1u);
    }
    else
    {
      *v38 = 93;
      ++*(_QWORD *)(v37 + 32);
    }
  }
  v24 = *a1;
  v25 = *(_WORD **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v25 <= 1u )
  {
    sub_CB6200(v24, (unsigned __int8 *)";\n", 2u);
  }
  else
  {
    *v25 = 2619;
    *(_QWORD *)(v24 + 32) += 2LL;
  }
  if ( v42 != (unsigned __int8 *)v44 )
    j_j___libc_free_0((unsigned __int64)v42);
}
