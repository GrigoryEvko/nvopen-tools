// Function: sub_24C9AA0
// Address: 0x24c9aa0
//
__int64 __fastcall sub_24C9AA0(__int64 a1, _QWORD *a2, const char *a3, _QWORD *a4)
{
  char v5; // bl
  unsigned int *v6; // rax
  size_t v7; // r12
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // r12
  char v11; // al
  unsigned int *v12; // rax
  size_t v13; // r8
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // r13
  char v17; // al
  _QWORD *v18; // rax
  bool v19; // zf
  _QWORD *v21; // rdi
  _BYTE *v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rax
  size_t n; // [rsp+8h] [rbp-158h]
  __int64 v29; // [rsp+48h] [rbp-118h]
  _BYTE *v30; // [rsp+50h] [rbp-110h] BYREF
  unsigned int *v31; // [rsp+58h] [rbp-108h]
  _QWORD v32[2]; // [rsp+60h] [rbp-100h] BYREF
  unsigned __int64 v33[2]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v34[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v35; // [rsp+90h] [rbp-D0h]
  unsigned int *v36[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE v37[16]; // [rsp+B0h] [rbp-B0h] BYREF
  __int16 v38; // [rsp+C0h] [rbp-A0h]
  __int64 v39; // [rsp+D0h] [rbp-90h]
  __int64 v40; // [rsp+D8h] [rbp-88h]
  __int16 v41; // [rsp+E0h] [rbp-80h]
  _QWORD *v42; // [rsp+E8h] [rbp-78h]
  void **v43; // [rsp+F0h] [rbp-70h]
  void **v44; // [rsp+F8h] [rbp-68h]
  __int64 v45; // [rsp+100h] [rbp-60h]
  int v46; // [rsp+108h] [rbp-58h]
  __int16 v47; // [rsp+10Ch] [rbp-54h]
  char v48; // [rsp+10Eh] [rbp-52h]
  __int64 v49; // [rsp+110h] [rbp-50h]
  __int64 v50; // [rsp+118h] [rbp-48h]
  void *v51; // [rsp+120h] [rbp-40h] BYREF
  void *v52; // [rsp+128h] [rbp-38h] BYREF

  v5 = 9 * (*(_DWORD *)(a1 + 604) != 1);
  v30 = v32;
  v6 = (unsigned int *)strlen(a3);
  v36[0] = v6;
  v7 = (size_t)v6;
  if ( (unsigned __int64)v6 > 0xF )
  {
    v30 = (_BYTE *)sub_22409D0((__int64)&v30, (unsigned __int64 *)v36, 0);
    v21 = v30;
    v32[0] = v36[0];
  }
  else
  {
    if ( v6 == (unsigned int *)1 )
    {
      LOBYTE(v32[0]) = *a3;
      v8 = v32;
      goto LABEL_4;
    }
    if ( !v6 )
    {
      v8 = v32;
      goto LABEL_4;
    }
    v21 = v32;
  }
  memcpy(v21, a3, v7);
  v6 = v36[0];
  v8 = v30;
LABEL_4:
  v31 = v6;
  *((_BYTE *)v6 + (_QWORD)v8) = 0;
  if ( *(_DWORD *)(a1 + 604) == 5 )
    sub_8FD6D0((__int64)v33, byte_42B6D7A, &v30);
  else
    sub_8FD6D0((__int64)v33, "__start___", &v30);
  v36[0] = (unsigned int *)v33;
  v38 = 260;
  BYTE4(v29) = 0;
  v9 = sub_BD2C40(88, unk_3F0FAE8);
  v10 = (__int64)v9;
  if ( v9 )
    sub_B30000((__int64)v9, (__int64)a2, a4, 0, v5, 0, (__int64)v36, 0, 0, v29, 0);
  if ( (_QWORD *)v33[0] != v34 )
    j_j___libc_free_0(v33[0]);
  if ( v30 != (_BYTE *)v32 )
    j_j___libc_free_0((unsigned __int64)v30);
  v11 = *(_BYTE *)(v10 + 32) & 0xCF | 0x10;
  *(_BYTE *)(v10 + 32) = v11;
  if ( (v11 & 0xF) != 9 )
    *(_BYTE *)(v10 + 33) |= 0x40u;
  v30 = v32;
  v12 = (unsigned int *)strlen(a3);
  v36[0] = v12;
  v13 = (size_t)v12;
  if ( (unsigned __int64)v12 > 0xF )
  {
    n = (size_t)v12;
    v22 = (_BYTE *)sub_22409D0((__int64)&v30, (unsigned __int64 *)v36, 0);
    v13 = n;
    v30 = v22;
    v23 = v22;
    v32[0] = v36[0];
  }
  else
  {
    if ( v12 == (unsigned int *)1 )
    {
      LOBYTE(v32[0]) = *a3;
      v14 = v32;
      goto LABEL_17;
    }
    if ( !v12 )
    {
      v14 = v32;
      goto LABEL_17;
    }
    v23 = v32;
  }
  memcpy(v23, a3, v13);
  v12 = v36[0];
  v14 = v30;
LABEL_17:
  v31 = v12;
  *((_BYTE *)v12 + (_QWORD)v14) = 0;
  if ( *(_DWORD *)(a1 + 604) == 5 )
    sub_8FD6D0((__int64)v33, byte_42B6D9E, &v30);
  else
    sub_8FD6D0((__int64)v33, "__stop___", &v30);
  v36[0] = (unsigned int *)v33;
  v38 = 260;
  BYTE4(v29) = 0;
  v15 = sub_BD2C40(88, unk_3F0FAE8);
  v16 = v15;
  if ( v15 )
    sub_B30000((__int64)v15, (__int64)a2, a4, 0, v5, 0, (__int64)v36, 0, 0, v29, 0);
  if ( (_QWORD *)v33[0] != v34 )
    j_j___libc_free_0(v33[0]);
  if ( v30 != (_BYTE *)v32 )
    j_j___libc_free_0((unsigned __int64)v30);
  v17 = v16[4] & 0xCF | 0x10;
  *((_BYTE *)v16 + 32) = v17;
  if ( (v17 & 0xF) != 9 )
    *((_BYTE *)v16 + 33) |= 0x40u;
  v18 = (_QWORD *)*a2;
  v43 = &v51;
  v42 = v18;
  v36[0] = (unsigned int *)v37;
  v36[1] = (unsigned int *)0x200000000LL;
  v51 = &unk_49DA100;
  v44 = &v52;
  v45 = 0;
  v52 = &unk_49DA0B0;
  v46 = 0;
  v19 = *(_DWORD *)(a1 + 604) == 1;
  v47 = 512;
  v48 = 7;
  v49 = 0;
  v50 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  if ( v19 )
  {
    v24 = *(_QWORD *)(a1 + 464);
    v35 = 257;
    v30 = (_BYTE *)sub_AD64C0(v24, 8, 0);
    v25 = sub_BCB2B0(v42);
    v10 = sub_921130(v36, v25, v10, &v30, 1, (__int64)v33, 0);
  }
  nullsub_61();
  v51 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v36[0] != v37 )
    _libc_free((unsigned __int64)v36[0]);
  return v10;
}
