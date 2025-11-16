// Function: sub_37FA9C0
// Address: 0x37fa9c0
//
_QWORD *__fastcall sub_37FA9C0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64); // rcx
  __int16 v10; // ax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  const void *v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdi
  size_t v25; // rdx
  size_t v26; // r15
  unsigned __int64 v27; // rdx
  size_t v28; // r15
  int v29; // eax
  char v30; // dl
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  const void *v37; // [rsp+8h] [rbp-228h]
  unsigned int v38; // [rsp+14h] [rbp-21Ch]
  _QWORD v39[2]; // [rsp+20h] [rbp-210h] BYREF
  _QWORD v40[2]; // [rsp+30h] [rbp-200h] BYREF
  _QWORD v41[8]; // [rsp+40h] [rbp-1F0h] BYREF
  _QWORD v42[4]; // [rsp+80h] [rbp-1B0h] BYREF
  char v43; // [rsp+A0h] [rbp-190h]
  _QWORD v44[2]; // [rsp+A8h] [rbp-188h] BYREF
  _QWORD v45[2]; // [rsp+B8h] [rbp-178h] BYREF
  _QWORD v46[3]; // [rsp+C8h] [rbp-168h] BYREF
  char *v47[3]; // [rsp+E0h] [rbp-150h] BYREF
  _BYTE v48[312]; // [rsp+F8h] [rbp-138h] BYREF

  v4 = (__int64)(a2 + 3);
  v8 = a2[1];
  v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 40LL);
  if ( (unsigned __int8)((*(_BYTE *)(a4 + 8) >> 5) - 2) <= 1u )
  {
    v10 = *(_WORD *)(a4 + 16);
    v11 = *(unsigned int *)(a4 + 2);
    LODWORD(v47[0]) = *(_DWORD *)(a4 + 12);
    WORD2(v47[0]) = v10;
    v38 = (unsigned int)v47[0];
    v12 = v9(v8, v11);
    v13 = a2[1];
    v39[0] = v12;
    v39[1] = v14;
    v40[0] = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v13 + 40LL))(v13, v38);
    v42[0] = "{0} {1}::*";
    v42[2] = v46;
    v40[1] = v15;
    v44[1] = v40;
    v44[0] = &unk_49DB108;
    v45[0] = &unk_49DB108;
    v45[1] = v39;
    v46[0] = v45;
    v46[1] = v44;
    v41[5] = 0x100000000LL;
    v43 = 1;
    v41[0] = &unk_49DD288;
    v42[1] = 10;
    v42[3] = 2;
    v47[0] = v48;
    v47[1] = 0;
    v47[2] = (char *)256;
    v41[1] = 2;
    memset(&v41[2], 0, 24);
    v41[6] = v47;
    sub_CB5980((__int64)v41, 0, 0, 0);
    sub_CB6840((__int64)v41, (__int64)v42);
    v41[0] = &unk_49DD388;
    sub_CB5840((__int64)v41);
    sub_37FA2C0(v4, v47, v16, v17, v18, v19);
    if ( v47[0] != v48 )
      _libc_free((unsigned __int64)v47[0]);
    goto LABEL_4;
  }
  v21 = (const void *)v9(v8, *(unsigned int *)(a4 + 2));
  v24 = a2[4];
  v26 = v25;
  v27 = v24 + v25;
  if ( v27 > a2[5] )
  {
    v37 = v21;
    sub_C8D290(v4, a2 + 6, v27, 1u, v22, v23);
    v24 = a2[4];
    v21 = v37;
  }
  if ( v26 )
  {
    memcpy((void *)(a2[3] + v24), v21, v26);
    v24 = a2[4];
  }
  v28 = v24 + v26;
  a2[4] = v28;
  v29 = *(_DWORD *)(a4 + 8);
  v30 = (unsigned __int8)v29 >> 5;
  if ( (unsigned __int8)v29 >> 5 == 1 )
  {
    if ( v28 + 1 > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v28 + 1, 1u, v22, v23);
      v28 = a2[4];
    }
    *(_BYTE *)(a2[3] + v28) = 38;
    ++a2[4];
    v29 = *(_DWORD *)(a4 + 8);
  }
  else if ( v30 == 4 )
  {
    if ( v28 + 2 > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v28 + 2, 1u, v22, v23);
      v28 = a2[4];
    }
    *(_WORD *)(a2[3] + v28) = 9766;
    a2[4] += 2LL;
    v29 = *(_DWORD *)(a4 + 8);
  }
  else if ( !v30 )
  {
    if ( v28 + 1 > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v28 + 1, 1u, v22, v23);
      v28 = a2[4];
    }
    *(_BYTE *)(a2[3] + v28) = 42;
    ++a2[4];
    v29 = *(_DWORD *)(a4 + 8);
  }
  if ( (v29 & 0x400) != 0 )
  {
    v35 = a2[4];
    if ( (unsigned __int64)(v35 + 6) > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v35 + 6, 1u, v22, v23);
      v35 = a2[4];
    }
    v36 = a2[3] + v35;
    *(_DWORD *)v36 = 1852793632;
    *(_WORD *)(v36 + 4) = 29811;
    a2[4] += 6LL;
    v29 = *(_DWORD *)(a4 + 8);
  }
  if ( (v29 & 0x200) != 0 )
  {
    v32 = a2[4];
    if ( (unsigned __int64)(v32 + 9) > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v32 + 9, 1u, v22, v23);
      v32 = a2[4];
    }
    v33 = a2[3] + v32;
    *(_QWORD *)v33 = 0x6C6974616C6F7620LL;
    *(_BYTE *)(v33 + 8) = 101;
    a2[4] += 9LL;
    v29 = *(_DWORD *)(a4 + 8);
    if ( (v29 & 0x800) == 0 )
      goto LABEL_15;
    goto LABEL_25;
  }
  if ( (v29 & 0x800) != 0 )
  {
LABEL_25:
    v34 = a2[4];
    if ( (unsigned __int64)(v34 + 12) > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v34 + 12, 1u, v22, v23);
      v34 = a2[4];
    }
    qmemcpy((void *)(a2[3] + v34), " __unaligned", 12);
    a2[4] += 12LL;
    v29 = *(_DWORD *)(a4 + 8);
  }
LABEL_15:
  if ( (v29 & 0x1000) != 0 )
  {
    v31 = a2[4];
    if ( (unsigned __int64)(v31 + 11) > a2[5] )
    {
      sub_C8D290(v4, a2 + 6, v31 + 11, 1u, v22, v23);
      v31 = a2[4];
    }
    qmemcpy((void *)(a2[3] + v31), " __restrict", 11);
    a2[4] += 11LL;
  }
LABEL_4:
  *a1 = 1;
  return a1;
}
