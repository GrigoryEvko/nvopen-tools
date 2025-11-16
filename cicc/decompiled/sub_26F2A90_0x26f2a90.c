// Function: sub_26F2A90
// Address: 0x26f2a90
//
__int64 __fastcall sub_26F2A90(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rbx
  unsigned __int8 *v4; // r14
  __int64 result; // rax
  __int64 v6; // r15
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  unsigned __int8 **v12; // rdx
  unsigned int v13; // r8d
  _QWORD *v14; // rax
  unsigned __int8 *v15; // rdi
  _BYTE *v16; // rsi
  __int64 *v17; // r14
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // eax
  int v22; // edi
  __int64 v23; // rax
  void *v24; // rdx
  void *v25; // rax
  int v26; // eax
  int v27; // esi
  __int64 v28; // rcx
  unsigned int v29; // eax
  unsigned __int8 *v30; // r8
  int v31; // r10d
  unsigned __int8 **v32; // r9
  int v33; // eax
  int v34; // eax
  __int64 v35; // r8
  int v36; // r10d
  unsigned int v37; // ecx
  unsigned __int8 *v38; // rsi
  unsigned int v39; // [rsp+Ch] [rbp-84h]
  const void *v40[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v41; // [rsp+20h] [rbp-70h] BYREF
  void *v42[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v43; // [rsp+50h] [rbp-40h]

  v3 = a3;
  v4 = *(unsigned __int8 **)(*(_QWORD *)(a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL);
  result = (unsigned int)*v4 - 5;
  if ( (unsigned __int8)(*v4 - 5) > 0x1Fu )
    return result;
  result = v4[1] & 0x7F;
  if ( (_BYTE)result != 1 )
    return result;
  v6 = *(_QWORD *)a1;
  v9 = *(_DWORD *)(*(_QWORD *)a1 + 24LL);
  if ( !v9 )
  {
    ++*(_QWORD *)v6;
    goto LABEL_31;
  }
  v10 = *(_QWORD *)(v6 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v14 = (_QWORD *)(v10 + 16LL * v13);
  v15 = (unsigned __int8 *)*v14;
  if ( v4 != (unsigned __int8 *)*v14 )
  {
    while ( v15 != (unsigned __int8 *)-4096LL )
    {
      if ( v15 == (unsigned __int8 *)-8192LL && !v12 )
        v12 = (unsigned __int8 **)v14;
      v13 = (v9 - 1) & (v11 + v13);
      v14 = (_QWORD *)(v10 + 16LL * v13);
      v15 = (unsigned __int8 *)*v14;
      if ( v4 == (unsigned __int8 *)*v14 )
        goto LABEL_6;
      ++v11;
    }
    if ( !v12 )
      v12 = (unsigned __int8 **)v14;
    v21 = *(_DWORD *)(v6 + 16);
    ++*(_QWORD *)v6;
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(v6 + 20) - v22 > v9 >> 3 )
      {
LABEL_24:
        *(_DWORD *)(v6 + 16) = v22;
        if ( *v12 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(v6 + 20);
        *v12 = v4;
        v17 = (__int64 *)(v12 + 1);
        v12[1] = 0;
        v6 = *(_QWORD *)a1;
        goto LABEL_27;
      }
      v39 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
      sub_AEB980(v6, v9);
      v33 = *(_DWORD *)(v6 + 24);
      if ( v33 )
      {
        v34 = v33 - 1;
        v35 = *(_QWORD *)(v6 + 8);
        v32 = 0;
        v36 = 1;
        v37 = v34 & v39;
        v22 = *(_DWORD *)(v6 + 16) + 1;
        v12 = (unsigned __int8 **)(v35 + 16LL * (v34 & v39));
        v38 = *v12;
        if ( v4 == *v12 )
          goto LABEL_24;
        while ( v38 != (unsigned __int8 *)-4096LL )
        {
          if ( v38 == (unsigned __int8 *)-8192LL && !v32 )
            v32 = v12;
          v37 = v34 & (v36 + v37);
          v12 = (unsigned __int8 **)(v35 + 16LL * v37);
          v38 = *v12;
          if ( v4 == *v12 )
            goto LABEL_24;
          ++v36;
        }
        goto LABEL_35;
      }
      goto LABEL_51;
    }
LABEL_31:
    sub_AEB980(v6, 2 * v9);
    v26 = *(_DWORD *)(v6 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(v6 + 8);
      v29 = (v26 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v22 = *(_DWORD *)(v6 + 16) + 1;
      v12 = (unsigned __int8 **)(v28 + 16LL * v29);
      v30 = *v12;
      if ( v4 == *v12 )
        goto LABEL_24;
      v31 = 1;
      v32 = 0;
      while ( v30 != (unsigned __int8 *)-4096LL )
      {
        if ( !v32 && v30 == (unsigned __int8 *)-8192LL )
          v32 = v12;
        v29 = v27 & (v31 + v29);
        v12 = (unsigned __int8 **)(v28 + 16LL * v29);
        v30 = *v12;
        if ( v4 == *v12 )
          goto LABEL_24;
        ++v31;
      }
LABEL_35:
      if ( v32 )
        v12 = v32;
      goto LABEL_24;
    }
LABEL_51:
    ++*(_DWORD *)(v6 + 16);
    BUG();
  }
LABEL_6:
  v16 = (_BYTE *)v14[1];
  v17 = v14 + 1;
  if ( !v16 )
  {
LABEL_27:
    v23 = *(_QWORD *)(a1 + 8);
    v24 = *(void **)v23;
    v25 = *(void **)(v23 + 8);
    LODWORD(v42[0]) = *(_DWORD *)(v6 + 16);
    v42[3] = v25;
    v42[2] = v24;
    v43 = 1289;
    sub_CA0F50((__int64 *)v40, v42);
    *v17 = sub_B9B140(**(__int64 ***)(a1 + 16), v40[0], (size_t)v40[1]);
    if ( v40[0] != &v41 )
      j_j___libc_free_0((unsigned __int64)v40[0]);
    v16 = (_BYTE *)*v17;
  }
  result = sub_B9F6F0(**(__int64 ***)(a1 + 16), v16);
  v18 = 32 * (v3 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + a2;
  if ( *(_QWORD *)v18 )
  {
    v19 = *(_QWORD *)(v18 + 8);
    **(_QWORD **)(v18 + 16) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
  }
  *(_QWORD *)v18 = result;
  if ( result )
  {
    v20 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v18 + 8) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = v18 + 8;
    *(_QWORD *)(v18 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v18;
  }
  return result;
}
