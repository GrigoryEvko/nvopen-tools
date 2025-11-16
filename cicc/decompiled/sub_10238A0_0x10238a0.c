// Function: sub_10238A0
// Address: 0x10238a0
//
__int64 __fastcall sub_10238A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v9; // r15
  unsigned int v10; // eax
  unsigned int v11; // r14d
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rsi
  int v16; // ecx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  _BYTE *v36; // r8
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  bool v42; // al
  __int64 v43; // [rsp+8h] [rbp-A8h]
  __int64 v44; // [rsp+10h] [rbp-A0h]
  __int64 v45; // [rsp+10h] [rbp-A0h]
  __int64 v48; // [rsp+28h] [rbp-88h]
  __int64 v49; // [rsp+28h] [rbp-88h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  _QWORD v51[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v52; // [rsp+40h] [rbp-70h]
  int v53; // [rsp+48h] [rbp-68h]
  __int64 v54; // [rsp+50h] [rbp-60h]
  __int64 v55; // [rsp+58h] [rbp-58h]
  char *v56[2]; // [rsp+60h] [rbp-50h] BYREF
  _BYTE v57[64]; // [rsp+70h] [rbp-40h] BYREF

  v9 = *(_QWORD *)(a1 + 8);
  LOBYTE(v10) = sub_D97040(a3, v9);
  if ( !(_BYTE)v10 )
    return 0;
  v11 = v10;
  if ( !a5 )
    a5 = sub_DD8400(a3, a1);
  if ( *((_WORD *)a5 + 12) != 8 || a2 != a5[6] )
    return 0;
  v48 = a5[6];
  v13 = sub_D4B130(v48);
  v14 = *(_QWORD *)(a1 - 8);
  v15 = v13;
  v16 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v16 )
  {
    v17 = 0;
    v18 = v14 + 32LL * *(unsigned int *)(a1 + 72);
    while ( v15 != *(_QWORD *)(v18 + 8 * v17) )
    {
      if ( v16 == (_DWORD)++v17 )
        goto LABEL_28;
    }
    v19 = 32 * v17;
  }
  else
  {
LABEL_28:
    v19 = 0x1FFFFFFFE0LL;
  }
  v44 = v48;
  v43 = *(_QWORD *)(v14 + v19);
  v49 = sub_D47930(a5[6]);
  if ( !v49 )
    return 0;
  v23 = sub_D33D80(a5, a3, v20, v21, v22);
  v24 = v49;
  v25 = v44;
  v26 = v23;
  if ( *(_WORD *)(v23 + 24) )
  {
    v45 = v49;
    v50 = v23;
    v42 = sub_DADE90(a3, v23, v25);
    v26 = v50;
    v24 = v45;
    if ( !v42 )
      return 0;
  }
  if ( *(_BYTE *)(v9 + 8) == 12 )
  {
    v33 = *(_QWORD *)(a1 - 8);
    v34 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
    {
      v35 = 0;
      do
      {
        if ( v24 == *(_QWORD *)(v33 + 32LL * *(unsigned int *)(a1 + 72) + 8 * v35) )
        {
          v34 = 32 * v35;
          goto LABEL_34;
        }
        ++v35;
      }
      while ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != (_DWORD)v35 );
      v34 = 0x1FFFFFFFE0LL;
    }
LABEL_34:
    v36 = *(_BYTE **)(v33 + v34);
    if ( (unsigned __int8)(*v36 - 42) >= 0x12u )
      v36 = 0;
    sub_1023480((__int64)v51, v43, 1, v26, (__int64)v36, a6);
    v40 = v52;
    v41 = *(_QWORD *)(a4 + 16);
    if ( v41 != v52 )
    {
      if ( v41 != 0 && v41 != -4096 && v41 != -8192 )
      {
        sub_BD60C0((_QWORD *)a4);
        v40 = v52;
      }
      LOBYTE(v37) = v40 != -4096;
      LOBYTE(v41) = v40 != 0;
      *(_QWORD *)(a4 + 16) = v40;
      if ( ((v40 != 0) & (unsigned __int8)v37) != 0 && v40 != -8192 )
        sub_BD6050((unsigned __int64 *)a4, v51[0] & 0xFFFFFFFFFFFFFFF8LL);
    }
    *(_DWORD *)(a4 + 24) = v53;
    *(_QWORD *)(a4 + 32) = v54;
    *(_QWORD *)(a4 + 40) = v55;
    sub_1021AD0(a4 + 48, v56, v41, v37, v38, v39);
    if ( v56[0] != v57 )
      _libc_free(v56[0], v56);
    v32 = v52;
    if ( v52 != -4096 && v52 != 0 )
      goto LABEL_26;
  }
  else
  {
    sub_1023480((__int64)v51, v43, 2, v26, 0, 0);
    v30 = v52;
    v31 = *(_QWORD *)(a4 + 16);
    if ( v31 != v52 )
    {
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
      {
        sub_BD60C0((_QWORD *)a4);
        v30 = v52;
      }
      LOBYTE(v27) = v30 != -4096;
      LOBYTE(v31) = v30 != 0;
      *(_QWORD *)(a4 + 16) = v30;
      if ( ((v30 != 0) & (unsigned __int8)v27) != 0 && v30 != -8192 )
        sub_BD6050((unsigned __int64 *)a4, v51[0] & 0xFFFFFFFFFFFFFFF8LL);
    }
    *(_DWORD *)(a4 + 24) = v53;
    *(_QWORD *)(a4 + 32) = v54;
    *(_QWORD *)(a4 + 40) = v55;
    sub_1021AD0(a4 + 48, v56, v31, v27, v28, v29);
    if ( v56[0] != v57 )
      _libc_free(v56[0], v56);
    v32 = v52;
    if ( v52 != 0 && v52 != -4096 )
    {
LABEL_26:
      if ( v32 != -8192 )
        sub_BD60C0(v51);
    }
  }
  return v11;
}
