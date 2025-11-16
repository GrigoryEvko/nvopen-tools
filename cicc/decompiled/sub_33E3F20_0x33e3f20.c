// Function: sub_33E3F20
// Address: 0x33e3f20
//
__int64 __fastcall sub_33E3F20(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5)
{
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int16 *v8; // rdx
  unsigned int v9; // r15d
  __int64 v10; // r12
  int v11; // eax
  void **v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // r13d
  __int64 v21; // r14
  unsigned int v22; // r15d
  __int64 v23; // rdx
  __int64 *v24; // rsi
  _BYTE *v25; // rax
  int v26; // eax
  __int64 v27; // rdx
  char v28; // cl
  __int64 v29; // rax
  __int64 v30; // rbx
  unsigned __int64 v31; // r12
  char v32; // [rsp+Ch] [rbp-114h]
  unsigned int v35; // [rsp+30h] [rbp-F0h]
  _BYTE *v36; // [rsp+30h] [rbp-F0h]
  unsigned __int8 v38; // [rsp+3Fh] [rbp-E1h]
  __int64 v39; // [rsp+40h] [rbp-E0h] BYREF
  int v40; // [rsp+48h] [rbp-D8h]
  __int64 v41; // [rsp+50h] [rbp-D0h]
  __int64 v42; // [rsp+58h] [rbp-C8h]
  _BYTE *v43; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+68h] [rbp-B8h]
  _BYTE v45[48]; // [rsp+70h] [rbp-B0h] BYREF
  void *v46; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-78h]
  _BYTE v48[48]; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+E0h] [rbp-40h]

  v38 = sub_33E22F0(a1);
  if ( !v38 )
    return v38;
  v8 = *(unsigned __int16 **)(a1 + 48);
  v9 = *(_DWORD *)(a1 + 64);
  v32 = a2;
  v10 = *((_QWORD *)v8 + 1);
  v11 = *v8;
  v44 = v10;
  LOWORD(v43) = v11;
  if ( (_WORD)v11 )
  {
    if ( (unsigned __int16)(v11 - 17) > 0xD3u )
    {
      LOWORD(v46) = v11;
      v47 = v10;
LABEL_6:
      if ( (_WORD)v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
        BUG();
      v12 = &v46;
      v35 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v11 - 16];
      LODWORD(v47) = v35;
      if ( v35 > 0x40 )
        goto LABEL_9;
      goto LABEL_13;
    }
    LOWORD(v11) = word_4456580[v11 - 1];
    v27 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v43) )
    {
      v47 = v10;
      LOWORD(v46) = 0;
      goto LABEL_12;
    }
    LOWORD(v11) = sub_3009970((__int64)&v43, a2, v13, v14, v15);
  }
  LOWORD(v46) = v11;
  v47 = v27;
  if ( (_WORD)v11 )
    goto LABEL_6;
LABEL_12:
  v41 = sub_3007260((__int64)&v46);
  v42 = v16;
  v35 = v41;
  LODWORD(v47) = v41;
  if ( (unsigned int)v41 > 0x40 )
  {
LABEL_9:
    sub_C43690((__int64)&v46, 0, 0);
    goto LABEL_14;
  }
LABEL_13:
  v46 = 0;
LABEL_14:
  v17 = v9;
  v43 = v45;
  v44 = 0x300000000LL;
  sub_33E3890((__int64)&v43, v9, (__int64)&v46, (__int64)v12, v5, v6);
  if ( (unsigned int)v47 > 0x40 && v46 )
    j_j___libc_free_0_0((unsigned __int64)v46);
  v20 = (v9 + 63) >> 6;
  v46 = v48;
  v47 = 0x600000000LL;
  if ( v20 > 6 )
  {
    sub_C8D5F0((__int64)&v46, v48, v20, 8u, v18, v19);
    memset(v46, 0, 8LL * v20);
    LODWORD(v47) = (v9 + 63) >> 6;
  }
  else
  {
    if ( v20 && 8LL * v20 )
      memset(v48, 0, 8LL * v20);
    LODWORD(v47) = (v9 + 63) >> 6;
  }
  v49 = v9;
  v21 = 0;
  if ( v9 )
  {
    v22 = v35;
    do
    {
      while ( 1 )
      {
        v23 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40 * v21);
        v26 = *(_DWORD *)(v23 + 24);
        if ( v26 != 51 )
          break;
        v28 = v21;
        v29 = (unsigned int)v21++ >> 6;
        *((_QWORD *)v46 + v29) |= 1LL << v28;
        if ( v21 == v17 )
          goto LABEL_42;
      }
      if ( v26 == 11 || v26 == 35 )
      {
        sub_C44740((__int64)&v39, (char **)(*(_QWORD *)(v23 + 96) + 24LL), v22);
      }
      else
      {
        if ( v26 != 12 && v26 != 36 )
          v23 = 0;
        v24 = (__int64 *)(*(_QWORD *)(v23 + 96) + 24LL);
        if ( (void *)*v24 == sub_C33340() )
          sub_C3E660((__int64)&v39, (__int64)v24);
        else
          sub_C3A850((__int64)&v39, v24);
      }
      v25 = &v43[16 * v21];
      if ( *((_DWORD *)v25 + 2) > 0x40u && *(_QWORD *)v25 )
      {
        v36 = &v43[16 * v21];
        j_j___libc_free_0_0(*(_QWORD *)v25);
        v25 = v36;
      }
      ++v21;
      *(_QWORD *)v25 = v39;
      *((_DWORD *)v25 + 2) = v40;
    }
    while ( v21 != v17 );
  }
LABEL_42:
  sub_33E3AB0(v32, a3, a4, (__int64)v43, (unsigned int)v44, a5, &v46);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  v30 = (__int64)v43;
  v31 = (unsigned __int64)&v43[16 * (unsigned int)v44];
  if ( v43 != (_BYTE *)v31 )
  {
    do
    {
      v31 -= 16LL;
      if ( *(_DWORD *)(v31 + 8) > 0x40u && *(_QWORD *)v31 )
        j_j___libc_free_0_0(*(_QWORD *)v31);
    }
    while ( v30 != v31 );
    v31 = (unsigned __int64)v43;
  }
  if ( (_BYTE *)v31 != v45 )
    _libc_free(v31);
  return v38;
}
