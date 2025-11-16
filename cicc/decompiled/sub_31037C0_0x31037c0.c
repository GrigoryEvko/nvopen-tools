// Function: sub_31037C0
// Address: 0x31037c0
//
__int64 __fastcall sub_31037C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v14; // r13
  int v15; // eax
  __int64 v16; // rcx
  int v17; // esi
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 *v26; // rdi
  bool v27; // r14
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 *v30; // rax
  int v31; // eax
  __int64 v32; // rbx
  __int64 v33; // rax
  int v34; // r8d
  __int64 v35; // [rsp+0h] [rbp-90h]
  __int64 v36; // [rsp+0h] [rbp-90h]
  __int64 v37; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+8h] [rbp-88h]
  __int64 *v40; // [rsp+10h] [rbp-80h] BYREF
  __int64 v41; // [rsp+18h] [rbp-78h]
  _BYTE v42[112]; // [rsp+20h] [rbp-70h] BYREF

  v4 = *(_QWORD *)(a2 + 72);
  if ( !*(_QWORD *)(a1 + 24)
    || (v5 = a1,
        a1 += 8,
        v6 = (*(__int64 (__fastcall **)(__int64, __int64))(v5 + 32))(a1, v4),
        v4 = *(_QWORD *)(a2 + 72),
        v7 = v6,
        !*(_QWORD *)(v5 + 56)) )
  {
    sub_4263D6(a1, v4, a3);
  }
  v8 = (*(__int64 (__fastcall **)(__int64, __int64))(v5 + 64))(v5 + 40, v4);
  if ( v8 )
  {
    v9 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    if ( *(_DWORD *)(v8 + 32) > (unsigned int)v9 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v9);
      if ( v10 )
      {
        v11 = *(_QWORD *)(v10 + 8);
        if ( v11 )
          return *(_QWORD *)v11;
      }
    }
  }
  v14 = 0;
  if ( !v7 )
    goto LABEL_14;
  v15 = *(_DWORD *)(v7 + 24);
  v16 = *(_QWORD *)(v7 + 8);
  if ( !v15 )
    goto LABEL_46;
  v17 = v15 - 1;
  v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (__int64 *)(v16 + 16LL * v18);
  v20 = *v19;
  if ( a2 != *v19 )
  {
    v31 = 1;
    while ( v20 != -4096 )
    {
      v34 = v31 + 1;
      v18 = v17 & (v31 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( a2 == *v19 )
        goto LABEL_12;
      v31 = v34;
    }
    v14 = 0;
    goto LABEL_46;
  }
LABEL_12:
  v14 = v19[1];
  if ( !v14 )
  {
LABEL_46:
    v7 = 0;
    goto LABEL_14;
  }
  v7 = **(_QWORD **)(v14 + 32);
LABEL_14:
  v12 = *(_QWORD *)(a2 + 16);
  v40 = (__int64 *)v42;
  v41 = 0x800000000LL;
  if ( !v12 )
    return v12;
  while ( 1 )
  {
    v21 = *(_QWORD *)(v12 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
      break;
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      return v12;
  }
  v22 = v14 + 56;
LABEL_18:
  v23 = *(_QWORD *)(v21 + 40);
  if ( a2 == v23 )
    goto LABEL_22;
  if ( v7 != a2 )
    goto LABEL_20;
  if ( !*(_BYTE *)(v14 + 84) )
  {
    v37 = v22;
    v35 = *(_QWORD *)(v21 + 40);
    v30 = sub_C8CA60(v22, v35);
    v22 = v37;
    if ( v30 )
      goto LABEL_22;
    v24 = (unsigned int)v41;
    v23 = v35;
    v25 = (unsigned int)v41 + 1LL;
    if ( v25 <= HIDWORD(v41) )
    {
LABEL_21:
      v40[v24] = v23;
      LODWORD(v41) = v41 + 1;
      goto LABEL_22;
    }
LABEL_41:
    v36 = v22;
    v38 = v23;
    sub_C8D5F0((__int64)&v40, v42, v25, 8u, v22, v23);
    v24 = (unsigned int)v41;
    v22 = v36;
    v23 = v38;
    goto LABEL_21;
  }
  v28 = *(_QWORD **)(v14 + 64);
  v29 = &v28[*(unsigned int *)(v14 + 76)];
  if ( v28 == v29 )
  {
LABEL_20:
    v24 = (unsigned int)v41;
    v25 = (unsigned int)v41 + 1LL;
    if ( v25 <= HIDWORD(v41) )
      goto LABEL_21;
    goto LABEL_41;
  }
  while ( v23 != *v28 )
  {
    if ( v29 == ++v28 )
      goto LABEL_20;
  }
LABEL_22:
  while ( 1 )
  {
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      break;
    v21 = *(_QWORD *)(v12 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
      goto LABEL_18;
  }
  v26 = v40;
  if ( (_DWORD)v41 )
  {
    if ( (_DWORD)v41 == 1 )
    {
      v12 = *v40;
    }
    else
    {
      v27 = 1;
      v12 = 0;
      if ( (_DWORD)v41 == 2 )
      {
        v32 = v40[1];
        v39 = *v40;
        v12 = sub_AA5510(*v40);
        v33 = sub_AA5510(v32);
        if ( v39 == v33 )
        {
          v26 = v40;
          v12 = v39;
          v27 = v39 == 0;
        }
        else
        {
          v26 = v40;
          if ( v32 == v12 || v12 == v33 )
            v27 = v12 == 0;
          else
            v12 = 0;
        }
      }
      if ( v14 && v27 )
        v12 = **(_QWORD **)(v14 + 32);
    }
  }
  else
  {
    v12 = 0;
  }
  if ( v26 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v26);
  return v12;
}
