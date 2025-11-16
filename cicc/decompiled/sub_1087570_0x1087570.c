// Function: sub_1087570
// Address: 0x1087570
//
__int64 __fastcall sub_1087570(__int64 a1, __int16 *a2)
{
  bool v3; // zf
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int16 v7; // ax
  __int16 v8; // ax
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rdi
  __int16 v23; // ax
  __int64 v24; // rdi
  int v25; // eax
  unsigned int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rdi
  unsigned int v30; // eax
  __int64 v31; // rdi
  __int16 v32; // ax
  __int64 v33; // rdi
  __int16 v34; // ax
  __int64 v35; // rdi
  unsigned __int8 v36[36]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *(_BYTE *)(a1 + 240) == 0;
  v4 = *(_QWORD *)(a1 + 8);
  if ( v3 )
  {
    v23 = *a2;
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v23 = __ROL2__(v23, 8);
    *(_WORD *)v36 = v23;
    sub_CB6200(v4, v36, 2u);
    v24 = *(_QWORD *)(a1 + 8);
    v25 = *((_DWORD *)a2 + 1);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      LOWORD(v25) = __ROL2__(v25, 8);
    *(_WORD *)v36 = v25;
    sub_CB6200(v24, v36, 2u);
    v26 = *((_DWORD *)a2 + 2);
    v27 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v26 = _byteswap_ulong(v26);
    *(_DWORD *)v36 = v26;
    sub_CB6200(v27, v36, 4u);
    v28 = *((_DWORD *)a2 + 3);
    v29 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v28 = _byteswap_ulong(v28);
    *(_DWORD *)v36 = v28;
    sub_CB6200(v29, v36, 4u);
    v30 = *((_DWORD *)a2 + 4);
    v31 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v30 = _byteswap_ulong(v30);
    *(_DWORD *)v36 = v30;
    sub_CB6200(v31, v36, 4u);
    v32 = a2[10];
    v33 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v32 = __ROL2__(v32, 8);
    *(_WORD *)v36 = v32;
    sub_CB6200(v33, v36, 2u);
    v34 = a2[11];
    v35 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v34 = __ROL2__(v34, 8);
    *(_WORD *)v36 = v34;
    return sub_CB6200(v35, v36, 2u);
  }
  else
  {
    *(_WORD *)v36 = 0;
    sub_CB6200(v4, v36, 2u);
    v5 = *(_QWORD *)(a1 + 8);
    *(_WORD *)v36 = -1;
    sub_CB6200(v5, v36, 2u);
    v6 = *(_QWORD *)(a1 + 8);
    v7 = 2;
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v7 = 512;
    *(_WORD *)v36 = v7;
    sub_CB6200(v6, v36, 2u);
    v8 = *a2;
    v9 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v8 = __ROL2__(v8, 8);
    *(_WORD *)v36 = v8;
    sub_CB6200(v9, v36, 2u);
    v10 = *((_DWORD *)a2 + 2);
    v11 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v10 = _byteswap_ulong(v10);
    *(_DWORD *)v36 = v10;
    sub_CB6200(v11, v36, 4u);
    sub_CB6200(*(_QWORD *)(a1 + 8), byte_3F8FC60, 0x10u);
    v12 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)v36 = 0;
    sub_CB6200(v12, v36, 4u);
    v13 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)v36 = 0;
    sub_CB6200(v13, v36, 4u);
    v14 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)v36 = 0;
    sub_CB6200(v14, v36, 4u);
    v15 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)v36 = 0;
    sub_CB6200(v15, v36, 4u);
    v16 = *((_DWORD *)a2 + 1);
    v17 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v16 = _byteswap_ulong(v16);
    *(_DWORD *)v36 = v16;
    sub_CB6200(v17, v36, 4u);
    v18 = *((_DWORD *)a2 + 3);
    v19 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v18 = _byteswap_ulong(v18);
    *(_DWORD *)v36 = v18;
    sub_CB6200(v19, v36, 4u);
    v20 = *((_DWORD *)a2 + 4);
    v21 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 16) != 1 )
      v20 = _byteswap_ulong(v20);
    *(_DWORD *)v36 = v20;
    return sub_CB6200(v21, v36, 4u);
  }
}
