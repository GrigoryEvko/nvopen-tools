// Function: sub_34A5A20
// Address: 0x34a5a20
//
bool __fastcall sub_34A5A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // r8
  unsigned int v12; // esi
  unsigned __int64 *v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rcx
  _BYTE *v25; // rdi
  unsigned int v27; // eax
  unsigned __int64 *v28; // r14
  unsigned __int64 *v29; // rsi
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r9
  __int64 v32; // rdx
  unsigned __int64 v33; // rdi
  unsigned __int64 *v34; // rax
  unsigned __int64 v35; // rax
  int v36; // esi
  unsigned __int64 *v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rcx
  int v43; // edi
  unsigned __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned __int64 v48; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v49; // [rsp+8h] [rbp-E8h]
  __int64 v50; // [rsp+10h] [rbp-E0h] BYREF
  _BYTE *v51; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v52; // [rsp+20h] [rbp-D0h]
  _BYTE v53[64]; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v54; // [rsp+68h] [rbp-88h] BYREF
  _BYTE *v55; // [rsp+70h] [rbp-80h] BYREF
  __int64 v56; // [rsp+78h] [rbp-78h]
  _BYTE v57[112]; // [rsp+80h] [rbp-70h] BYREF

  v8 = a2 + 8;
  v9 = *(unsigned int *)(a2 + 204);
  if ( (_DWORD)v9 )
  {
    v10 = *(unsigned int *)(a1 + 200);
    v11 = *(_QWORD *)(a2 + 8);
    v50 = a1 + 8;
    v51 = v53;
    v52 = 0x400000000LL;
    if ( !(_DWORD)v10 )
    {
      v12 = *(_DWORD *)(a1 + 204);
      if ( v12 )
      {
        v13 = (unsigned __int64 *)(a1 + 16);
        while ( *v13 < v11 )
        {
          v10 = (unsigned int)(v10 + 1);
          v13 += 2;
          if ( v12 == (_DWORD)v10 )
            goto LABEL_8;
        }
        v12 = v10;
      }
LABEL_8:
      sub_34A26E0((__int64)&v50, v12, v10, v9, v11, a6);
      v17 = (unsigned int)v52;
      if ( !(_DWORD)v52 )
        goto LABEL_9;
      goto LABEL_20;
    }
    sub_34A3C90((__int64)&v50, v11, v10, v9, v11, a6);
  }
  else
  {
    v47 = *(_DWORD *)(a1 + 204);
    v50 = a1 + 8;
    v51 = v53;
    v52 = 0x400000000LL;
    sub_34A26E0((__int64)&v50, v47, a3, v9, a5, a6);
  }
  v17 = (unsigned int)v52;
  if ( !(_DWORD)v52 )
  {
LABEL_9:
    v54 = v8;
    v18 = *(unsigned int *)(a2 + 204);
    v55 = v57;
    v56 = 0x400000000LL;
LABEL_10:
    sub_34A26E0((__int64)&v54, v18, v17, v14, v15, v16);
    goto LABEL_11;
  }
LABEL_20:
  v14 = *((unsigned int *)v51 + 2);
  if ( *((_DWORD *)v51 + 3) >= (unsigned int)v14 )
    goto LABEL_9;
  v15 = *(_QWORD *)sub_34A2590((__int64)&v50);
  v54 = v8;
  v56 = 0x400000000LL;
  v27 = *(_DWORD *)(a2 + 200);
  v55 = v57;
  if ( !v27 )
  {
    v18 = *(unsigned int *)(a2 + 204);
    if ( (_DWORD)v18 )
    {
      v28 = (unsigned __int64 *)(a2 + 16);
      while ( v15 > *v28 )
      {
        ++v27;
        v28 += 2;
        if ( (_DWORD)v18 == v27 )
          goto LABEL_10;
      }
      v18 = v27;
    }
    goto LABEL_10;
  }
  v18 = v15;
  sub_34A3C90((__int64)&v54, v15, v17, v14, v15, v16);
LABEL_11:
  sub_34A5870((__int64)&v50, v18, v19, v20, v21, v22);
  v23 = (unsigned int)v52;
  if ( (_DWORD)v52 )
  {
    while ( 1 )
    {
      v24 = (__int64)v51;
      v25 = v55;
      if ( *((_DWORD *)v51 + 3) >= *((_DWORD *)v51 + 2) || !(_DWORD)v56 || *((_DWORD *)v55 + 3) >= *((_DWORD *)v55 + 2) )
        break;
      v29 = (unsigned __int64 *)(*(_QWORD *)&v51[16 * v23 - 16] + 16LL * *(unsigned int *)&v51[16 * v23 - 4]);
      v30 = v29[1];
      v31 = *v29;
      v32 = (__int64)&v55[16 * (unsigned int)v56 - 16];
      v33 = *(unsigned int *)(a3 + 12);
      v34 = (unsigned __int64 *)(*(_QWORD *)v32 + 16LL * *(unsigned int *)(v32 + 12));
      if ( v34[1] <= v30 )
        v30 = v34[1];
      if ( *v34 >= v31 )
        v31 = *v34;
      v35 = *(unsigned int *)(a3 + 8);
      v36 = *(_DWORD *)(a3 + 8);
      if ( v35 >= v33 )
      {
        if ( v33 < v35 + 1 )
        {
          v48 = v30;
          v49 = v31;
          sub_C8D5F0(a3, (const void *)(a3 + 16), v35 + 1, 0x10u, v30, v31);
          v35 = *(unsigned int *)(a3 + 8);
          v30 = v48;
          v31 = v49;
        }
        v44 = (unsigned __int64 *)(*(_QWORD *)a3 + 16 * v35);
        *v44 = v31;
        v44[1] = v30;
        v45 = (unsigned int)v56;
        v46 = (__int64)v55;
        ++*(_DWORD *)(a3 + 8);
        v38 = v50;
        v24 = (__int64)v51;
        v32 = v46 + 16 * v45 - 16;
      }
      else
      {
        v37 = (unsigned __int64 *)(*(_QWORD *)a3 + 16 * v35);
        if ( v37 )
        {
          *v37 = v31;
          v37[1] = v30;
          v36 = *(_DWORD *)(a3 + 8);
          v24 = (__int64)v51;
          v32 = (__int64)&v55[16 * (unsigned int)v56 - 16];
        }
        v38 = v50;
        *(_DWORD *)(a3 + 8) = v36 + 1;
      }
      v39 = v24 + 16LL * (unsigned int)v52 - 16;
      v40 = *(unsigned int *)(v32 + 12);
      v41 = *(unsigned int *)(v39 + 12);
      v42 = *(_QWORD *)v39 + 16 * v41;
      if ( *(_QWORD *)(v42 + 8) <= *(_QWORD *)(*(_QWORD *)v32 + 16 * v40 + 8) )
      {
        v41 = (unsigned int)(v41 + 1);
        *(_DWORD *)(v39 + 12) = v41;
        if ( (_DWORD)v41 == *(_DWORD *)&v51[16 * (unsigned int)v52 - 8] )
        {
          v41 = *(unsigned int *)(v38 + 192);
          if ( (_DWORD)v41 )
            sub_F03D40((__int64 *)&v51, v41);
        }
      }
      else
      {
        v43 = v40 + 1;
        *(_DWORD *)(v32 + 12) = v43;
        if ( v43 == *(_DWORD *)&v55[16 * (unsigned int)v56 - 8] )
        {
          v41 = *(unsigned int *)(v54 + 192);
          if ( (_DWORD)v41 )
            sub_F03D40((__int64 *)&v55, v41);
        }
      }
      sub_34A5870((__int64)&v50, v41, v32, v42, v38, v39);
      v23 = (unsigned int)v52;
      if ( !(_DWORD)v52 )
        goto LABEL_40;
    }
  }
  else
  {
LABEL_40:
    v25 = v55;
  }
  if ( v25 != v57 )
    _libc_free((unsigned __int64)v25);
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  return *(_DWORD *)(a3 + 8) != 0;
}
