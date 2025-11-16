// Function: sub_34A3D10
// Address: 0x34a3d10
//
__int64 __fastcall sub_34A3D10(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  _QWORD *v9; // rbx
  unsigned int v10; // eax
  unsigned int v11; // esi
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  unsigned int v15; // esi
  _BYTE **v16; // r9
  char *v17; // rdi
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  char *v20; // rax
  int v21; // esi
  __int64 v22; // rcx
  bool v23; // al
  _BYTE *v25; // r8
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // edx
  char *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // [rsp+10h] [rbp-170h]
  __int64 v34; // [rsp+18h] [rbp-168h]
  bool v35; // [rsp+18h] [rbp-168h]
  __int64 v36; // [rsp+18h] [rbp-168h]
  int v37; // [rsp+18h] [rbp-168h]
  __int64 v38; // [rsp+20h] [rbp-160h] BYREF
  _BYTE *v39; // [rsp+28h] [rbp-158h] BYREF
  __int64 v40; // [rsp+30h] [rbp-150h]
  _BYTE v41[72]; // [rsp+38h] [rbp-148h] BYREF
  __int64 v42; // [rsp+80h] [rbp-100h]
  _BYTE *v43; // [rsp+88h] [rbp-F8h] BYREF
  __int64 v44; // [rsp+90h] [rbp-F0h]
  _BYTE v45[72]; // [rsp+98h] [rbp-E8h] BYREF
  __int64 v46; // [rsp+E0h] [rbp-A0h] BYREF
  char *v47; // [rsp+E8h] [rbp-98h] BYREF
  __int64 v48; // [rsp+F0h] [rbp-90h]
  _BYTE v49[64]; // [rsp+F8h] [rbp-88h] BYREF
  int v50; // [rsp+138h] [rbp-48h]
  unsigned __int64 v51; // [rsp+140h] [rbp-40h]
  __int64 v52; // [rsp+148h] [rbp-38h]

  v7 = a2 + 8;
  v9 = a3;
  v40 = 0x400000000LL;
  v10 = *(_DWORD *)(a2 + 200);
  v38 = a2 + 8;
  v39 = v41;
  if ( v10 )
  {
    sub_34A3C90((__int64)&v38, (unsigned __int64)a3, (__int64)a3, a2, a5, a6);
    v14 = a2;
  }
  else
  {
    v11 = *(_DWORD *)(a2 + 204);
    if ( v11 )
    {
      a3 = (_QWORD *)(a2 + 16);
      while ( (unsigned __int64)v9 > *a3 )
      {
        ++v10;
        a3 += 2;
        if ( v11 == v10 )
          goto LABEL_7;
      }
      v11 = v10;
    }
LABEL_7:
    v34 = a2;
    sub_34A26E0((__int64)&v38, v11, (__int64)a3, a2, a5, a6);
    v14 = v34;
  }
  v15 = *(_DWORD *)(v14 + 204);
  v46 = v7;
  v48 = 0x400000000LL;
  v47 = v49;
  sub_34A26E0((__int64)&v46, v15, 0x400000000LL, v14, v12, v13);
  if ( !(_DWORD)v40 || *((_DWORD *)v39 + 3) >= *((_DWORD *)v39 + 2) )
  {
    v17 = v47;
    v23 = 1;
    if ( (_DWORD)v48 )
      v23 = *((_DWORD *)v47 + 3) >= *((_DWORD *)v47 + 2);
    if ( v47 == v49 )
      goto LABEL_17;
LABEL_16:
    v35 = v23;
    _libc_free((unsigned __int64)v17);
    v23 = v35;
LABEL_17:
    v22 = a1 + 24;
    if ( v23 )
    {
      *(_QWORD *)(a1 + 8) = v22;
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 16) = 0x400000000LL;
      *(_QWORD *)(a1 + 96) = 0;
      *(_QWORD *)(a1 + 104) = 0;
      *(_DWORD *)(a1 + 88) = -1;
      goto LABEL_19;
    }
    v21 = v40;
    goto LABEL_24;
  }
  v17 = v47;
  v18 = 0x400000000LL;
  v19 = (unsigned __int64)&v39[16 * (unsigned int)v40 - 16];
  v20 = &v47[16 * (unsigned int)v48 - 16];
  if ( *(_DWORD *)(v19 + 12) == *((_DWORD *)v20 + 3) )
  {
    v23 = *(_QWORD *)v19 == *(_QWORD *)v20;
    if ( v47 == v49 )
      goto LABEL_17;
    goto LABEL_16;
  }
  if ( v47 == v49 )
  {
    v22 = a1 + 24;
    v44 = 0x400000000LL;
    v43 = v45;
    v42 = v38;
    goto LABEL_38;
  }
  _libc_free((unsigned __int64)v47);
  v21 = v40;
  v22 = a1 + 24;
LABEL_24:
  v18 = v38;
  v25 = v45;
  v43 = v45;
  v42 = v38;
  v44 = 0x400000000LL;
  if ( !v21 )
  {
    v46 = v38;
    v47 = v49;
    v48 = 0x400000000LL;
    v51 = 0;
    v52 = 0;
    goto LABEL_26;
  }
LABEL_38:
  v33 = v22;
  sub_349DB40((__int64)&v43, (__int64)&v39, v18, v22, (__int64)v45, (__int64)&v43);
  v21 = v44;
  v47 = v49;
  v16 = &v43;
  v22 = v33;
  v46 = v42;
  v25 = v45;
  v48 = 0x400000000LL;
  if ( (_DWORD)v44 )
  {
    sub_349DB40((__int64)&v47, (__int64)&v43, v31, v33, (__int64)v45, (__int64)&v43);
    v21 = v48;
    v50 = 0;
    v51 = 0;
    v22 = v33;
    v52 = 0;
    v25 = v45;
    if ( (_DWORD)v48 && *((_DWORD *)v47 + 3) < *((_DWORD *)v47 + 2) )
    {
      v37 = v48;
      v51 = *(_QWORD *)sub_34A2590((__int64)&v46);
      v32 = (__int64 *)sub_34A25B0((__int64)&v46);
      v21 = v37;
      v22 = v33;
      v25 = v45;
      v52 = *v32;
      goto LABEL_27;
    }
  }
  else
  {
    v51 = 0;
    v52 = 0;
  }
LABEL_26:
  v50 = -1;
LABEL_27:
  if ( v43 != v45 )
  {
    v36 = v22;
    _libc_free((unsigned __int64)v43);
    v21 = v48;
    v22 = v36;
  }
  v26 = v51;
  if ( (unsigned __int64)v9 >= v51 )
    v50 = (_DWORD)v9 - v51;
  v27 = v46;
  *(_QWORD *)(a1 + 8) = v22;
  *(_QWORD *)(a1 + 16) = 0x400000000LL;
  *(_QWORD *)a1 = v27;
  if ( v21 )
  {
    sub_349DC20(a1 + 8, &v47, v27, v22, (__int64)v25, (__int64)v16);
    v26 = v51;
  }
  v28 = v50;
  v29 = v47;
  *(_QWORD *)(a1 + 96) = v26;
  v30 = v52;
  *(_DWORD *)(a1 + 88) = v28;
  *(_QWORD *)(a1 + 104) = v30;
  if ( v29 != v49 )
    _libc_free((unsigned __int64)v29);
LABEL_19:
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  return a1;
}
