// Function: sub_108F000
// Address: 0x108f000
//
void __fastcall sub_108F000(__int64 a1, __int64 a2)
{
  char *v3; // rbx
  __int64 v4; // r14
  char v5; // al
  unsigned __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rdi
  unsigned __int32 v9; // r8d
  unsigned __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rdi
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rdi
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rdi
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rdi
  unsigned __int64 v25; // r8
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // edx
  __int64 v29; // rdi
  unsigned __int32 v30; // r8d
  __int64 v31; // rdi
  unsigned int v32; // eax
  __int64 v33; // rdi
  unsigned __int64 v34; // r8
  __int16 v35; // r8
  __int16 v36; // ax
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rdi
  unsigned __int32 v40; // r8d
  unsigned __int32 v41; // r8d
  unsigned __int32 v42; // r8d
  unsigned __int32 v43; // r8d
  int v44; // [rsp-54h] [rbp-54h]
  _QWORD v45[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_WORD *)(a2 + 56) != 0xFFFD )
  {
    v3 = (char *)(a2 + 8);
    v4 = *(_QWORD *)(a1 + 168);
    v44 = *(_DWORD *)(a2 + 52);
    do
    {
      v5 = *v3++;
      LOBYTE(v45[0]) = v5;
      sub_CB6200(v4, (unsigned __int8 *)v45, 1u);
    }
    while ( (char *)(a2 + 16) != v3 );
    v6 = 0;
    if ( (v44 & 0x10) == 0 )
      v6 = *(_QWORD *)(a2 + 16);
    v7 = *(_DWORD *)(a1 + 176);
    v8 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v34 = _byteswap_uint64(v6);
      if ( v7 != 1 )
        v6 = v34;
      v45[0] = v6;
      sub_CB6200(v8, (unsigned __int8 *)v45, 8u);
    }
    else
    {
      v9 = _byteswap_ulong(v6);
      if ( v7 != 1 )
        LODWORD(v6) = v9;
      LODWORD(v45[0]) = v6;
      sub_CB6200(v8, (unsigned __int8 *)v45, 4u);
    }
    v10 = 0;
    if ( (v44 & 0x8010) == 0 )
      v10 = *(_QWORD *)(a2 + 16);
    v11 = *(_DWORD *)(a1 + 176);
    v12 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v13 = _byteswap_uint64(v10);
      if ( v11 != 1 )
        v10 = v13;
      v45[0] = v10;
      sub_CB6200(v12, (unsigned __int8 *)v45, 8u);
    }
    else
    {
      v40 = _byteswap_ulong(v10);
      if ( v11 != 1 )
        LODWORD(v10) = v40;
      LODWORD(v45[0]) = v10;
      sub_CB6200(v12, (unsigned __int8 *)v45, 4u);
    }
    v14 = *(_QWORD *)(a2 + 24);
    v15 = *(_DWORD *)(a1 + 176);
    v16 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v17 = _byteswap_uint64(v14);
      if ( v15 != 1 )
        v14 = v17;
      v45[0] = v14;
      sub_CB6200(v16, (unsigned __int8 *)v45, 8u);
    }
    else
    {
      v43 = _byteswap_ulong(v14);
      if ( v15 != 1 )
        LODWORD(v14) = v43;
      LODWORD(v45[0]) = v14;
      sub_CB6200(v16, (unsigned __int8 *)v45, 4u);
    }
    v18 = *(_QWORD *)(a2 + 32);
    v19 = *(_DWORD *)(a1 + 176);
    v20 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v21 = _byteswap_uint64(v18);
      if ( v19 != 1 )
        v18 = v21;
      v45[0] = v18;
      sub_CB6200(v20, (unsigned __int8 *)v45, 8u);
    }
    else
    {
      v42 = _byteswap_ulong(v18);
      if ( v19 != 1 )
        LODWORD(v18) = v42;
      LODWORD(v45[0]) = v18;
      sub_CB6200(v20, (unsigned __int8 *)v45, 4u);
    }
    v22 = *(_QWORD *)(a2 + 40);
    v23 = *(_DWORD *)(a1 + 176);
    v24 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v25 = _byteswap_uint64(v22);
      if ( v23 != 1 )
        v22 = v25;
      v45[0] = v22;
      sub_CB6200(v24, (unsigned __int8 *)v45, 8u);
    }
    else
    {
      v41 = _byteswap_ulong(v22);
      if ( v23 != 1 )
        LODWORD(v22) = v41;
      LODWORD(v45[0]) = v22;
      sub_CB6200(v24, (unsigned __int8 *)v45, 4u);
    }
    v26 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v45[0] = 0;
      sub_CB6200(v26, (unsigned __int8 *)v45, 8u);
    }
    else
    {
      LODWORD(v45[0]) = 0;
      sub_CB6200(v26, (unsigned __int8 *)v45, 4u);
    }
    v27 = *(_DWORD *)(a2 + 48);
    v28 = *(_DWORD *)(a1 + 176);
    v29 = *(_QWORD *)(a1 + 168);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    {
      v30 = _byteswap_ulong(v27);
      if ( v28 != 1 )
        v27 = v30;
      LODWORD(v45[0]) = v27;
      sub_CB6200(v29, (unsigned __int8 *)v45, 4u);
      v31 = *(_QWORD *)(a1 + 168);
      LODWORD(v45[0]) = 0;
      sub_CB6200(v31, (unsigned __int8 *)v45, 4u);
      v32 = *(_DWORD *)(a2 + 52);
      v33 = *(_QWORD *)(a1 + 168);
      if ( *(_DWORD *)(a1 + 176) != 1 )
        v32 = _byteswap_ulong(v32);
      LODWORD(v45[0]) = v32;
      sub_CB6200(v33, (unsigned __int8 *)v45, 4u);
      sub_CB6C70(*(_QWORD *)(a1 + 168), 4u);
    }
    else
    {
      v35 = __ROL2__(v27, 8);
      if ( v28 != 1 )
        LOWORD(v27) = v35;
      LOWORD(v45[0]) = v27;
      sub_CB6200(v29, (unsigned __int8 *)v45, 2u);
      if ( (v44 & 0x8000) != 0 || (v36 = 0, *(_DWORD *)(a2 + 48) == 0xFFFF) )
        v36 = *(_DWORD *)(a2 + 48);
      v37 = *(_QWORD *)(a1 + 168);
      if ( *(_DWORD *)(a1 + 176) != 1 )
        v36 = __ROL2__(v36, 8);
      LOWORD(v45[0]) = v36;
      sub_CB6200(v37, (unsigned __int8 *)v45, 2u);
      v38 = *(_DWORD *)(a2 + 52);
      v39 = *(_QWORD *)(a1 + 168);
      if ( *(_DWORD *)(a1 + 176) != 1 )
        v38 = _byteswap_ulong(v38);
      LODWORD(v45[0]) = v38;
      sub_CB6200(v39, (unsigned __int8 *)v45, 4u);
    }
  }
}
