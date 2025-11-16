// Function: sub_392E3E0
// Address: 0x392e3e0
//
__int64 __fastcall sub_392E3E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  _DWORD *v5; // rdx
  __int64 v6; // rdi
  char v7; // dl
  _BYTE *v8; // rax
  __int64 v9; // rdi
  char v10; // al
  _BYTE *v11; // rdx
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // rdi
  char v15; // si
  char *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rdi
  __int16 v22; // ax
  __int16 v23; // cx
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rdi
  unsigned __int32 v30; // ecx
  __int64 v31; // rdi
  int v32; // eax
  __int16 v33; // cx
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // rdi
  int v37; // eax
  __int16 v38; // cx
  __int64 v39; // rdi
  int v40; // eax
  __int64 v41; // rdi
  __int16 v42; // cx
  char v44[40]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_DWORD **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 3u )
  {
    sub_16E7EE0(v4, byte_4530676, 4u);
  }
  else
  {
    *v5 = 1179403647;
    *(_QWORD *)(v4 + 24) += 4LL;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 2 - ((*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0);
  v8 = *(_BYTE **)(v6 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v6 + 16) )
  {
    sub_16E7DE0(v6, 2 - ((*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0));
  }
  else
  {
    *(_QWORD *)(v6 + 24) = v8 + 1;
    *v8 = v7;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = (*(_DWORD *)(a1 + 16) != 1) + 1;
  v11 = *(_BYTE **)(v9 + 24);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v9 + 16) )
  {
    sub_16E7DE0(v9, (*(_DWORD *)(a1 + 16) != 1) + 1);
  }
  else
  {
    *(_QWORD *)(v9 + 24) = v11 + 1;
    *v11 = v10;
  }
  v12 = *(_QWORD *)(a1 + 8);
  v13 = *(_BYTE **)(v12 + 24);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 16) )
  {
    sub_16E7DE0(v12, 1);
  }
  else
  {
    *(_QWORD *)(v12 + 24) = v13 + 1;
    *v13 = 1;
  }
  v14 = *(_QWORD *)(a1 + 8);
  v15 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8LL);
  v16 = *(char **)(v14 + 24);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v14 + 16) )
  {
    sub_16E7DE0(v14, v15);
  }
  else
  {
    *(_QWORD *)(v14 + 24) = v16 + 1;
    *v16 = v15;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(_BYTE **)(v17 + 24);
  if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 16) )
  {
    sub_16E7DE0(v17, 0);
  }
  else
  {
    *(_QWORD *)(v17 + 24) = v18 + 1;
    *v18 = 0;
  }
  sub_16E8900(*(_QWORD *)(a1 + 8), 7u);
  v19 = *(_QWORD *)(a1 + 8);
  v20 = -((unsigned int)(*(_DWORD *)(a1 + 16) - 1) < 2);
  LOBYTE(v20) = (unsigned int)(*(_DWORD *)(a1 + 16) - 1) < 2;
  *(_WORD *)v44 = v20 + 256;
  sub_16E7EE0(v19, v44, 2u);
  v21 = *(_QWORD *)(a1 + 8);
  v22 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 10LL);
  v23 = __ROL2__(v22, 8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    v22 = v23;
  *(_WORD *)v44 = v22;
  sub_16E7EE0(v21, v44, 2u);
  v24 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)v44 = (unsigned int)(*(_DWORD *)(a1 + 16) - 1) < 2 ? 1 : 0x1000000;
  sub_16E7EE0(v24, v44, 4u);
  v25 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    *(_QWORD *)v44 = 0;
    sub_16E7EE0(v25, v44, 8u);
  }
  else
  {
    *(_DWORD *)v44 = 0;
    sub_16E7EE0(v25, v44, 4u);
  }
  v26 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    *(_QWORD *)v44 = 0;
    sub_16E7EE0(v26, v44, 8u);
  }
  else
  {
    *(_DWORD *)v44 = 0;
    sub_16E7EE0(v26, v44, 4u);
  }
  v27 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) != 0 )
  {
    *(_QWORD *)v44 = 0;
    sub_16E7EE0(v27, v44, 8u);
  }
  else
  {
    *(_DWORD *)v44 = 0;
    sub_16E7EE0(v27, v44, 4u);
  }
  v28 = *(_DWORD *)(a2 + 488);
  v29 = *(_QWORD *)(a1 + 8);
  v30 = _byteswap_ulong(v28);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    v28 = v30;
  *(_DWORD *)v44 = v28;
  sub_16E7EE0(v29, v44, 4u);
  v31 = *(_QWORD *)(a1 + 8);
  v32 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0 ? 52 : 64;
  v33 = __ROL2__((*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0 ? 52 : 64, 8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    LOWORD(v32) = v33;
  *(_WORD *)v44 = v32;
  sub_16E7EE0(v31, v44, 2u);
  v34 = *(_QWORD *)(a1 + 8);
  *(_WORD *)v44 = 0;
  sub_16E7EE0(v34, v44, 2u);
  v35 = *(_QWORD *)(a1 + 8);
  *(_WORD *)v44 = 0;
  sub_16E7EE0(v35, v44, 2u);
  v36 = *(_QWORD *)(a1 + 8);
  v37 = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0 ? 40 : 64;
  v38 = __ROL2__((*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 12LL) & 2) == 0 ? 40 : 64, 8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    LOWORD(v37) = v38;
  *(_WORD *)v44 = v37;
  sub_16E7EE0(v36, v44, 2u);
  v39 = *(_QWORD *)(a1 + 8);
  *(_WORD *)v44 = 0;
  sub_16E7EE0(v39, v44, 2u);
  v40 = *(_DWORD *)(a1 + 92);
  v41 = *(_QWORD *)(a1 + 8);
  v42 = __ROL2__(v40, 8);
  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    LOWORD(v40) = v42;
  *(_WORD *)v44 = v40;
  return sub_16E7EE0(v41, v44, 2u);
}
