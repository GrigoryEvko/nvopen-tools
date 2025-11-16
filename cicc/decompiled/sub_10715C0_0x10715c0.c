// Function: sub_10715C0
// Address: 0x10715c0
//
__int64 __fastcall sub_10715C0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        unsigned int a8)
{
  __int64 v13; // rax
  __int64 v14; // rdi
  size_t v15; // rdx
  unsigned __int8 *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // eax
  unsigned int v29; // ecx
  __int64 *v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rdi
  __int64 result; // rax
  __int64 v36; // rdi
  unsigned __int32 v37; // eax
  __int64 v38; // rdi
  unsigned __int32 v39; // eax
  __int64 v40; // rdi
  int v41; // edx
  int v42; // r8d
  unsigned __int64 v43; // [rsp+8h] [rbp-48h]
  unsigned __int8 v44[56]; // [rsp+18h] [rbp-38h] BYREF

  v13 = sub_E5CAC0(a2, a3);
  v14 = *(_QWORD *)(a1 + 2048);
  v43 = v13;
  if ( (*(_BYTE *)(a3 + 48) & 0x20) != 0 )
    a5 = 0;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 80LL))(v14);
  sub_1071270(a1, *(unsigned __int8 **)(a3 + 128), *(_QWORD *)(a3 + 136), 16);
  v15 = 16;
  v16 = (unsigned __int8 *)(a3 + 148);
  if ( !*(_BYTE *)(a3 + 163) )
  {
    v16 = (unsigned __int8 *)(a3 + 148);
    v15 = strlen((const char *)(a3 + 148));
  }
  sub_1071270(a1, v16, v15, 16);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 104) + 8LL) & 1) != 0 )
  {
    v17 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      a4 = _byteswap_uint64(a4);
    *(_QWORD *)v44 = a4;
    sub_CB6200(v17, v44, 8u);
    v18 = *(_QWORD *)(a1 + 2048);
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v43 = _byteswap_uint64(v43);
    *(_QWORD *)v44 = v43;
    sub_CB6200(v18, v44, 8u);
  }
  else
  {
    v36 = *(_QWORD *)(a1 + 2048);
    v37 = a4;
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v37 = _byteswap_ulong(a4);
    *(_DWORD *)v44 = v37;
    sub_CB6200(v36, v44, 4u);
    v38 = *(_QWORD *)(a1 + 2048);
    v39 = v43;
    if ( *(_DWORD *)(a1 + 2056) != 1 )
      v39 = _byteswap_ulong(v43);
    *(_DWORD *)v44 = v39;
    sub_CB6200(v38, v44, 4u);
  }
  v19 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a5 = _byteswap_ulong(a5);
  *(_DWORD *)v44 = a5;
  sub_CB6200(v19, v44, 4u);
  v20 = *(unsigned __int8 *)(a3 + 32);
  v21 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v20 = _byteswap_ulong(v20);
  *(_DWORD *)v44 = v20;
  sub_CB6200(v21, v44, 4u);
  v22 = *(_QWORD *)(a1 + 2048);
  v23 = 0;
  if ( a8 )
    v23 = a7;
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v23 = _byteswap_ulong(v23);
  *(_DWORD *)v44 = v23;
  sub_CB6200(v22, v44, 4u);
  v24 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a8 = _byteswap_ulong(a8);
  *(_DWORD *)v44 = a8;
  sub_CB6200(v24, v44, 4u);
  v25 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    a6 = _byteswap_ulong(a6);
  *(_DWORD *)v44 = a6;
  sub_CB6200(v25, v44, 4u);
  v26 = *(_DWORD *)(a1 + 192);
  v27 = *(_QWORD *)(a1 + 176);
  if ( v26 )
  {
    v28 = v26 - 1;
    v29 = v28 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v30 = (__int64 *)(v27 + 16LL * v29);
    v31 = *v30;
    if ( a3 == *v30 )
    {
LABEL_25:
      v26 = *((_DWORD *)v30 + 2);
    }
    else
    {
      v41 = 1;
      while ( v31 != -4096 )
      {
        v42 = v41 + 1;
        v29 = v28 & (v41 + v29);
        v30 = (__int64 *)(v27 + 16LL * v29);
        v31 = *v30;
        if ( a3 == *v30 )
          goto LABEL_25;
        v41 = v42;
      }
      v26 = 0;
    }
  }
  v32 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v26 = _byteswap_ulong(v26);
  *(_DWORD *)v44 = v26;
  sub_CB6200(v32, v44, 4u);
  v33 = *(_DWORD *)(a3 + 168);
  v34 = *(_QWORD *)(a1 + 2048);
  if ( *(_DWORD *)(a1 + 2056) != 1 )
    v33 = _byteswap_ulong(v33);
  *(_DWORD *)v44 = v33;
  sub_CB6200(v34, v44, 4u);
  result = *(_QWORD *)(a1 + 104);
  if ( (*(_BYTE *)(result + 8) & 1) != 0 )
  {
    v40 = *(_QWORD *)(a1 + 2048);
    *(_DWORD *)v44 = 0;
    return sub_CB6200(v40, v44, 4u);
  }
  return result;
}
