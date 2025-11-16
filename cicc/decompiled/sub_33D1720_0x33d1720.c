// Function: sub_33D1720
// Address: 0x33d1720
//
__int64 __fastcall sub_33D1720(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  __int64 v6; // r12
  int i; // eax
  __int64 v8; // rsi
  int v9; // r13d
  _DWORD *v10; // rax
  int v11; // ebx
  __int64 v12; // r15
  int v13; // edx
  __int64 v15; // rax
  unsigned __int16 *v16; // rdx
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rdx
  unsigned int v20; // eax
  unsigned int v21; // ebx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // esi
  unsigned int v29; // eax
  unsigned int v30; // ebx
  _DWORD *v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // [rsp+8h] [rbp-98h]
  unsigned int v34; // [rsp+8h] [rbp-98h]
  int v35; // [rsp+10h] [rbp-90h]
  int v36; // [rsp+1Ch] [rbp-84h]
  __int16 v37; // [rsp+20h] [rbp-80h] BYREF
  __int64 v38; // [rsp+28h] [rbp-78h]
  __int16 v39; // [rsp+30h] [rbp-70h] BYREF
  __int64 v40; // [rsp+38h] [rbp-68h]
  unsigned __int64 v41; // [rsp+40h] [rbp-60h] BYREF
  __int64 v42; // [rsp+48h] [rbp-58h]
  unsigned __int64 v43; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-48h]
  char v45; // [rsp+60h] [rbp-40h]

  v6 = a1;
  for ( i = *(_DWORD *)(a1 + 24); i == 234; i = *(_DWORD *)(v6 + 24) )
    v6 = **(_QWORD **)(v6 + 40);
  v8 = a2 ^ 1u;
  LOBYTE(v5) = v8 & (i == 168);
  if ( (_BYTE)v5 )
  {
    v44 = 1;
    v43 = 0;
    LOBYTE(v20) = sub_33D1410(v6, (__int64)&v43, a3, a4, a5);
    v5 = v20;
    if ( (_BYTE)v20 )
    {
      v21 = v44;
      if ( !v44 )
        return v5;
      if ( v44 <= 0x40 )
      {
        LOBYTE(v5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v44) == v43;
        return v5;
      }
      LOBYTE(v5) = v21 == (unsigned int)sub_C445E0((__int64)&v43);
    }
    else if ( v44 <= 0x40 )
    {
      return v5;
    }
LABEL_29:
    if ( v43 )
      j_j___libc_free_0_0(v43);
    return v5;
  }
  if ( i != 156 )
    return v5;
  v9 = *(_DWORD *)(v6 + 64);
  if ( !v9 )
    return v5;
  v10 = *(_DWORD **)(v6 + 40);
  v11 = 0;
  while ( 1 )
  {
    v12 = *(_QWORD *)v10;
    v13 = *(_DWORD *)(*(_QWORD *)v10 + 24LL);
    if ( v13 != 51 )
      break;
    ++v11;
    v10 += 10;
    if ( v9 == v11 )
      return v5;
  }
  v36 = v10[2];
  if ( v13 == 11 || v13 == 35 )
  {
    v15 = *(_QWORD *)(v12 + 96);
    v44 = *(_DWORD *)(v15 + 32);
    if ( v44 > 0x40 )
    {
      v8 = v15 + 24;
      sub_C43780((__int64)&v43, (const void **)(v15 + 24));
    }
    else
    {
      v43 = *(_QWORD *)(v15 + 24);
    }
  }
  else
  {
    if ( v13 != 12 && v13 != 36 )
      return v5;
    v8 = *(_QWORD *)(v12 + 96) + 24LL;
    if ( *(void **)v8 == sub_C33340() )
      sub_C3E660((__int64)&v41, v8);
    else
      sub_C3A850((__int64)&v41, (__int64 *)v8);
    v44 = v42;
    v43 = v41;
  }
  v45 = 1;
  v16 = *(unsigned __int16 **)(v6 + 48);
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  v37 = v17;
  v38 = v18;
  if ( (_WORD)v17 )
  {
    if ( (unsigned __int16)(v17 - 17) > 0xD3u )
    {
      v39 = v17;
      v40 = v18;
      goto LABEL_19;
    }
    LOWORD(v17) = word_4456580[v17 - 1];
    v32 = 0;
  }
  else
  {
    v33 = v18;
    if ( !sub_30070B0((__int64)&v37) )
    {
      v40 = v33;
      v39 = 0;
LABEL_33:
      v24 = sub_3007260((__int64)&v39);
      v25 = v19;
      v41 = v24;
      LODWORD(v19) = v24;
      v42 = v25;
      goto LABEL_34;
    }
    LOWORD(v17) = sub_3009970((__int64)&v37, v8, v33, v22, v23);
  }
  v39 = v17;
  v40 = v32;
  if ( !(_WORD)v17 )
    goto LABEL_33;
LABEL_19:
  if ( (_WORD)v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
    BUG();
  v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v17 - 16];
LABEL_34:
  v26 = v44;
  if ( v44 > 0x40 )
  {
    v34 = v44;
    v35 = v19;
    v29 = sub_C445E0((__int64)&v43);
    v26 = v34;
    LODWORD(v19) = v35;
  }
  else
  {
    _RAX = ~v43;
    __asm { tzcnt   rdi, rax }
    v29 = 64;
    if ( v43 != -1 )
      v29 = _RDI;
  }
  if ( (unsigned int)v19 > v29 )
  {
    if ( !v45 )
      return v5;
    v45 = 0;
    if ( v26 <= 0x40 )
      return v5;
    goto LABEL_29;
  }
  if ( v45 )
  {
    v45 = 0;
    if ( v26 > 0x40 )
    {
      if ( v43 )
        j_j___libc_free_0_0(v43);
    }
  }
  v30 = v11 + 1;
  if ( v9 == v30 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v31 = (_DWORD *)(*(_QWORD *)(v6 + 40) + 40LL * v30);
      if ( (v12 != *(_QWORD *)v31 || v31[2] != v36) && *(_DWORD *)(*(_QWORD *)v31 + 24LL) != 51 )
        break;
      if ( v9 == ++v30 )
        return 1;
    }
  }
  return v5;
}
