// Function: sub_1CB6E90
// Address: 0x1cb6e90
//
__int64 __fastcall sub_1CB6E90(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  unsigned __int8 v8; // al
  unsigned int v9; // eax
  unsigned int v10; // r15d
  __int64 *v11; // r12
  __int64 v13; // rax
  __int64 v14; // r8
  int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // r10
  size_t v18; // r15
  void *v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // r8
  _QWORD *v22; // r9
  _QWORD *v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // ecx
  __int64 v30; // r8
  signed __int64 v31; // r13
  _QWORD *v32; // r15
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // [rsp+0h] [rbp-40h]
  __int64 v39; // [rsp+0h] [rbp-40h]
  int v40; // [rsp+0h] [rbp-40h]
  int v41; // [rsp+8h] [rbp-38h]
  _QWORD *v42; // [rsp+8h] [rbp-38h]
  unsigned __int8 v43; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    while ( 1 )
    {
      v8 = *(_BYTE *)(a2 + 16);
      if ( v8 <= 0x17u )
        break;
      if ( (unsigned __int8)(v8 - 71) > 1u )
        goto LABEL_12;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v11 = *(__int64 **)(a2 - 8);
      else
        v11 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      a2 = *v11;
    }
    if ( v8 != 5 )
      goto LABEL_13;
    LOBYTE(v9) = sub_1594510(a2);
    v10 = v9;
    if ( !(_BYTE)v9 )
      break;
    a2 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
      return 0;
  }
  if ( (unsigned __int8)sub_1594530(a2) )
  {
    v28 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) == 15 )
    {
      v43 = sub_1CB6E90(a1, v28, a3, a4);
      if ( v43 )
      {
        v29 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v30 = (unsigned int)(v29 - 1);
        v40 = v29;
        if ( v29 == 1 )
        {
          v36 = *(_QWORD *)(a1 + 280);
          v37 = sub_16348C0(a2);
          *a4 += sub_15A9FF0(v36, v37, 0, 0);
        }
        else
        {
          v31 = 8 * v30;
          v32 = (_QWORD *)sub_22077B0(8 * v30);
          memset(v32, 0, v31);
          v33 = 1;
          do
          {
            v32[v33 - 1] = *(_QWORD *)(a2 + 24 * (v33 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
            ++v33;
          }
          while ( v33 != (unsigned int)(v40 - 2) + 2LL );
          v34 = *(_QWORD *)(a1 + 280);
          v35 = sub_16348C0(a2);
          *a4 += sub_15A9FF0(v34, v35, v32, v31 >> 3);
          j_j___libc_free_0(v32, v31);
        }
        return v43;
      }
    }
    return v10;
  }
  v8 = *(_BYTE *)(a2 + 16);
LABEL_12:
  if ( v8 != 56 )
  {
LABEL_13:
    *a3 = a2;
    v10 = 1;
    *a4 = 0;
    return v10;
  }
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v14 = (unsigned int)(v13 - 1);
  v15 = v13 - 1;
  v16 = -3 * v13;
  v17 = *(_QWORD *)(a2 + 8 * v16);
  if ( v14 )
  {
    v18 = 8 * v14;
    v38 = *(_QWORD *)(a2 + 8 * v16);
    v41 = v14;
    v19 = (void *)sub_22077B0(8 * v14);
    v20 = memset(v19, 0, v18);
    v15 = v41;
    v17 = v38;
    v21 = v18;
    v22 = v20;
  }
  else
  {
    v21 = 0;
    v22 = 0;
  }
  v23 = v22;
  LODWORD(v24) = 0;
  while ( (_DWORD)v24 != v15 )
  {
    ++v23;
    v24 = (unsigned int)(v24 + 1);
    v25 = *(_QWORD *)(a2 + 24 * (v24 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    *(v23 - 1) = v25;
    if ( *(_BYTE *)(v25 + 16) != 13 )
      goto LABEL_22;
  }
  v39 = v21;
  v42 = v22;
  v26 = sub_1CB6E90(a1, v17, a3, a4);
  v22 = v42;
  v21 = v39;
  v10 = v26;
  if ( !(_BYTE)v26 )
  {
LABEL_22:
    v10 = 0;
    goto LABEL_23;
  }
  v27 = sub_15A9FF0(*(_QWORD *)(a1 + 280), *(_QWORD *)(a2 + 56), v42, v39 >> 3);
  v21 = v39;
  *a4 += v27;
  v22 = v42;
LABEL_23:
  if ( v22 )
    j_j___libc_free_0(v22, v21);
  return v10;
}
