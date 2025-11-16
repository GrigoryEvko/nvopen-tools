// Function: sub_B4F0B0
// Address: 0xb4f0b0
//
__int64 __fastcall sub_B4F0B0(int *a1, __int64 a2, int a3, int *a4, int *a5)
{
  int *v5; // r8
  int *v6; // r9
  int v7; // r10d
  int *v9; // rax
  char v10; // cl
  char v11; // di
  int v12; // edx
  unsigned int v13; // r13d
  __int64 v15; // r14
  __int64 v16; // rcx
  char v17; // dl
  int v18; // eax
  __int64 v19; // rax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // r15d
  unsigned int v23; // r14d
  signed int v24; // eax
  int v25; // eax
  int v26; // ebx
  unsigned int v27; // eax
  unsigned int v28; // edi
  unsigned __int64 v29; // rax
  unsigned int v30; // eax
  signed int v31; // ebx
  unsigned __int64 v32; // rcx
  unsigned int v33; // eax
  unsigned int v38; // eax
  int v39; // [rsp+0h] [rbp-90h]
  int v40; // [rsp+0h] [rbp-90h]
  int v41; // [rsp+0h] [rbp-90h]
  int v42; // [rsp+0h] [rbp-90h]
  int v43; // [rsp+0h] [rbp-90h]
  int *v44; // [rsp+8h] [rbp-88h]
  int *v45; // [rsp+8h] [rbp-88h]
  int *v46; // [rsp+8h] [rbp-88h]
  int *v47; // [rsp+8h] [rbp-88h]
  int *v48; // [rsp+8h] [rbp-88h]
  char v49; // [rsp+10h] [rbp-80h]
  char v50; // [rsp+10h] [rbp-80h]
  char v51; // [rsp+10h] [rbp-80h]
  int v52; // [rsp+10h] [rbp-80h]
  char v53; // [rsp+10h] [rbp-80h]
  int *v54; // [rsp+10h] [rbp-80h]
  signed int v55; // [rsp+18h] [rbp-78h]
  __int64 v58; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-58h]
  unsigned __int64 v60; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v61; // [rsp+48h] [rbp-48h]
  unsigned __int64 v62; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v63; // [rsp+58h] [rbp-38h]

  if ( (int)a2 < a3 )
    return 0;
  v5 = &a1[a2];
  v6 = a1;
  v7 = a2;
  if ( a1 != v5 )
  {
    v9 = a1;
    v10 = 0;
    v11 = 0;
    do
    {
      v12 = *v9;
      if ( *v9 != -1 )
      {
        v11 |= a3 <= v12;
        v10 |= a3 > v12;
        if ( v10 )
        {
          if ( v11 )
            goto LABEL_12;
        }
      }
      ++v9;
    }
    while ( v5 != v9 );
    if ( v10 || v11 )
      return 0;
  }
LABEL_12:
  v59 = a2;
  if ( (unsigned int)a2 > 0x40 )
  {
    v54 = v6;
    sub_C43690(&v58, 0, 0);
    v61 = a2;
    sub_C43690(&v60, 0, 0);
    v63 = a2;
    sub_C43690(&v62, 0, 0);
    v7 = a2;
    v6 = v54;
  }
  else
  {
    v58 = 0;
    v61 = a2;
    v60 = 0;
    v63 = a2;
    v62 = 0;
    if ( !(_DWORD)a2 )
    {
      v23 = 0;
      v22 = 0;
      v17 = 1;
      v38 = 64;
      v13 = 1;
      goto LABEL_64;
    }
  }
  v15 = (unsigned int)(v7 - 1);
  v16 = 0;
  v17 = 1;
  v13 = 1;
  while ( 1 )
  {
    v20 = v6[v16];
    v21 = 1LL << v16;
    if ( v20 < 0 )
    {
      if ( v59 > 0x40 )
        *(_QWORD *)(v58 + 8LL * ((unsigned int)v16 >> 6)) |= v21;
      else
        v58 |= v21;
      goto LABEL_17;
    }
    if ( a3 > v20 )
      break;
    if ( v63 <= 0x40 )
      v62 |= v21;
    else
      *(_QWORD *)(v62 + 8LL * ((unsigned int)v16 >> 6)) |= v21;
    v18 = a3 + v16;
    LOBYTE(v18) = a3 + (_DWORD)v16 == v20;
    v13 &= v18;
LABEL_17:
    v19 = v16 + 1;
    if ( v16 == v15 )
      goto LABEL_26;
LABEL_18:
    v16 = v19;
  }
  if ( v61 > 0x40 )
    *(_QWORD *)(v60 + 8LL * ((unsigned int)v16 >> 6)) |= v21;
  else
    v60 |= v21;
  v17 &= v20 == (_DWORD)v16;
  v19 = v16 + 1;
  if ( v16 != v15 )
    goto LABEL_18;
LABEL_26:
  v22 = v61;
  v23 = v63;
  if ( v61 > 0x40 )
  {
    v39 = v7;
    v44 = v6;
    v49 = v17;
    v24 = sub_C44590(&v60);
    v7 = v39;
    v6 = v44;
    v55 = v24;
    v17 = v49;
    goto LABEL_28;
  }
  _RAX = v60;
  __asm { tzcnt   rsi, rax }
  v38 = 64;
  if ( v60 )
    v38 = _RSI;
LABEL_64:
  if ( v22 <= v38 )
    v38 = v22;
  v55 = v38;
LABEL_28:
  if ( v23 <= 0x40 )
  {
    _RAX = v62;
    v26 = 64;
    __asm { tzcnt   rsi, rax }
    if ( v62 )
      v26 = _RSI;
    if ( v26 > v23 )
      v26 = v23;
  }
  else
  {
    v40 = v7;
    v45 = v6;
    v50 = v17;
    v25 = sub_C44590(&v62);
    v7 = v40;
    v6 = v45;
    v17 = v50;
    v26 = v25;
  }
  if ( v22 > 0x40 )
  {
    v41 = v7;
    v46 = v6;
    v51 = v17;
    v27 = sub_C444A0(&v60);
    v7 = v41;
    v6 = v46;
    v17 = v51;
    v22 = v27;
    goto LABEL_32;
  }
  if ( v60 )
  {
    _BitScanReverse64(&v32, v60);
    v22 = v22 - 64 + (v32 ^ 0x3F);
    if ( v23 > 0x40 )
    {
LABEL_55:
      v43 = v7;
      v48 = v6;
      v53 = v17;
      v33 = sub_C444A0(&v62);
      v7 = v43;
      v6 = v48;
      v17 = v53;
      v28 = v33;
      goto LABEL_35;
    }
  }
  else
  {
LABEL_32:
    if ( v23 > 0x40 )
      goto LABEL_55;
  }
  v28 = v23;
  if ( v62 )
  {
    _BitScanReverse64(&v29, v62);
    v28 = v23 - 64 + (v29 ^ 0x3F);
  }
LABEL_35:
  if ( v17
    && (v42 = v7, v47 = v6, v52 = v7 - v28 - v26, v30 = sub_B487F0(&v6[v26], v52, a3), v6 = v47, v7 = v42, (_BYTE)v30) )
  {
    v13 = v30;
    *a4 = v52;
    *a5 = v26;
  }
  else if ( (_BYTE)v13 )
  {
    v31 = v7 - v22 - v55;
    v13 = sub_B487F0(&v6[v55], v31, a3);
    if ( (_BYTE)v13 )
    {
      *a4 = v31;
      *a5 = v55;
    }
  }
  if ( v23 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  return v13;
}
