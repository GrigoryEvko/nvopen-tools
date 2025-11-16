// Function: sub_299E0C0
// Address: 0x299e0c0
//
_QWORD *__fastcall sub_299E0C0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rbx
  unsigned __int8 *v5; // rdx
  _BYTE *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  const char *v14; // r15
  size_t v15; // rax
  size_t v16; // r9
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // rdi
  size_t v19; // rbx
  const void *v20; // r12
  void *v21; // rdi
  size_t n; // [rsp+28h] [rbp-928h]
  __int64 v25; // [rsp+38h] [rbp-918h]
  unsigned __int8 *v26; // [rsp+40h] [rbp-910h] BYREF
  size_t v27; // [rsp+48h] [rbp-908h]
  _QWORD v28[2]; // [rsp+50h] [rbp-900h] BYREF
  char *v29; // [rsp+60h] [rbp-8F0h] BYREF
  size_t v30; // [rsp+68h] [rbp-8E8h]
  _BYTE v31[16]; // [rsp+70h] [rbp-8E0h] BYREF
  _QWORD v32[3]; // [rsp+80h] [rbp-8D0h] BYREF
  _BYTE *v33; // [rsp+98h] [rbp-8B8h]
  _BYTE *v34; // [rsp+A0h] [rbp-8B0h]
  __int64 v35; // [rsp+A8h] [rbp-8A8h]
  unsigned __int64 *v36; // [rsp+B0h] [rbp-8A0h]
  unsigned __int64 v37[8]; // [rsp+C0h] [rbp-890h] BYREF
  unsigned __int64 v38[3]; // [rsp+100h] [rbp-850h] BYREF
  _BYTE v39[2104]; // [rsp+118h] [rbp-838h] BYREF

  v38[0] = (unsigned __int64)v39;
  v36 = v38;
  v35 = 0x100000000LL;
  v32[0] = &unk_49DD288;
  v38[1] = 0;
  v38[2] = 2048;
  v32[1] = 2;
  v32[2] = 0;
  v33 = 0;
  v34 = 0;
  sub_CB5980((__int64)v32, 0, 0, 0);
  sub_CB59D0((__int64)v32, *((unsigned int *)a2 + 2));
  v4 = *a2;
  v25 = *a2 + 56LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v25 )
  {
    while ( 1 )
    {
      v14 = *(const char **)v4;
      v26 = (unsigned __int8 *)v28;
      if ( !v14 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v15 = strlen(v14);
      v37[0] = v15;
      v16 = v15;
      if ( v15 > 0xF )
        break;
      if ( v15 == 1 )
      {
        LOBYTE(v28[0]) = *v14;
        v5 = (unsigned __int8 *)v28;
      }
      else
      {
        if ( v15 )
        {
          v18 = (unsigned __int8 *)v28;
          goto LABEL_26;
        }
        v5 = (unsigned __int8 *)v28;
      }
LABEL_4:
      v27 = v15;
      v5[v15] = 0;
      if ( !*(_DWORD *)(v4 + 48) )
        goto LABEL_5;
      if ( v27 == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v26, ":", 1u);
      v29 = v31;
      v37[5] = 0x100000000LL;
      v30 = 0;
      v37[0] = (unsigned __int64)&unk_49DD210;
      v37[6] = (unsigned __int64)&v29;
      v31[0] = 0;
      memset(&v37[1], 0, 32);
      sub_CB5980((__int64)v37, 0, 0, 0);
      sub_CB59D0((__int64)v37, *(unsigned int *)(v4 + 48));
      v37[0] = (unsigned __int64)&unk_49DD210;
      sub_CB5840((__int64)v37);
      sub_2241490((unsigned __int64 *)&v26, v29, v30);
      if ( v29 == v31 )
      {
LABEL_5:
        v6 = v34;
        if ( v33 == v34 )
          goto LABEL_24;
      }
      else
      {
        j_j___libc_free_0((unsigned __int64)v29);
        v6 = v34;
        if ( v33 == v34 )
        {
LABEL_24:
          v7 = (_QWORD *)sub_CB6200((__int64)v32, (unsigned __int8 *)" ", 1u);
          goto LABEL_7;
        }
      }
      *v6 = 32;
      v7 = v32;
      ++v34;
LABEL_7:
      v8 = sub_CB59D0((__int64)v7, *(_QWORD *)(v4 + 40));
      v9 = *(_BYTE **)(v8 + 32);
      if ( *(_BYTE **)(v8 + 24) == v9 )
      {
        v8 = sub_CB6200(v8, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v9 = 32;
        ++*(_QWORD *)(v8 + 32);
      }
      v10 = sub_CB59D0(v8, *(_QWORD *)(v4 + 8));
      v11 = *(_BYTE **)(v10 + 32);
      if ( *(_BYTE **)(v10 + 24) == v11 )
      {
        v10 = sub_CB6200(v10, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v11 = 32;
        ++*(_QWORD *)(v10 + 32);
      }
      v12 = sub_CB59D0(v10, v27);
      v13 = *(_BYTE **)(v12 + 32);
      if ( *(_BYTE **)(v12 + 24) == v13 )
      {
        v12 = sub_CB6200(v12, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v13 = 32;
        ++*(_QWORD *)(v12 + 32);
      }
      sub_CB6200(v12, v26, v27);
      if ( v26 != (unsigned __int8 *)v28 )
        j_j___libc_free_0((unsigned __int64)v26);
      v4 += 56;
      if ( v25 == v4 )
        goto LABEL_30;
    }
    n = v15;
    v17 = (unsigned __int8 *)sub_22409D0((__int64)&v26, v37, 0);
    v16 = n;
    v26 = v17;
    v18 = v17;
    v28[0] = v37[0];
LABEL_26:
    memcpy(v18, v14, v16);
    v15 = v37[0];
    v5 = v26;
    goto LABEL_4;
  }
LABEL_30:
  v19 = v36[1];
  v20 = (const void *)*v36;
  v21 = a1 + 3;
  a1[1] = 0;
  *a1 = a1 + 3;
  a1[2] = 64;
  if ( v19 > 0x40 )
  {
    sub_C8D290((__int64)a1, v21, v19, 1u, v2, v3);
    v21 = (void *)(*a1 + a1[1]);
  }
  else if ( !v19 )
  {
    goto LABEL_32;
  }
  memcpy(v21, v20, v19);
  v19 += a1[1];
LABEL_32:
  a1[1] = v19;
  v32[0] = &unk_49DD388;
  sub_CB5840((__int64)v32);
  if ( (_BYTE *)v38[0] != v39 )
    _libc_free(v38[0]);
  return a1;
}
