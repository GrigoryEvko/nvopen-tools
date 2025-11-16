// Function: sub_28FBEB0
// Address: 0x28fbeb0
//
void __fastcall sub_28FBEB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // r8
  _QWORD *v12; // rax
  int v13; // ecx
  __int64 v14; // r13
  __int64 v15; // rax
  bool v16; // bl
  int v17; // edx
  int v18; // edi
  __int64 v19; // rsi
  unsigned int v20; // ecx
  _QWORD *v21; // rdx
  __int64 v22; // r8
  _QWORD *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // edx
  int v27; // esi
  __int64 v28; // rdi
  unsigned int v29; // edx
  _QWORD *v30; // rcx
  __int64 v31; // r8
  _QWORD *v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // edx
  int v37; // ecx
  __int64 v38; // rdi
  unsigned int v39; // edx
  _QWORD *v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // rax
  _QWORD *v43; // r14
  __int64 v44; // rax
  _BYTE *v45; // rdi
  _QWORD *v46; // r14
  _QWORD *v47; // rbx
  _BYTE *v48; // rax
  int v49; // r8d
  int v50; // edx
  int v51; // ecx
  int v52; // r9d
  int v53; // r9d
  _QWORD *v54; // [rsp+0h] [rbp-F0h]
  _QWORD *v55; // [rsp+0h] [rbp-F0h]
  __int64 v56; // [rsp+0h] [rbp-F0h]
  __int64 v57; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+18h] [rbp-D8h]
  __int64 v59; // [rsp+20h] [rbp-D0h]
  _QWORD v60[3]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD *v61; // [rsp+48h] [rbp-A8h]
  __int64 v62[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v63; // [rsp+60h] [rbp-90h]
  __int64 v64; // [rsp+70h] [rbp-80h] BYREF
  __int64 v65; // [rsp+78h] [rbp-78h]
  __int64 v66; // [rsp+80h] [rbp-70h]
  _QWORD *v67; // [rsp+88h] [rbp-68h]
  _BYTE *v68; // [rsp+90h] [rbp-60h] BYREF
  __int64 v69; // [rsp+98h] [rbp-58h]
  _BYTE v70[80]; // [rsp+A0h] [rbp-50h] BYREF

  v9 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v10 = *(_QWORD **)(a2 - 8);
    v11 = (__int64)&v10[(unsigned __int64)v9 / 8];
  }
  else
  {
    v11 = a2;
    v10 = (_QWORD *)(a2 - v9);
  }
  v12 = v70;
  v13 = 0;
  v14 = v9 >> 5;
  v68 = v70;
  v69 = 0x400000000LL;
  if ( (unsigned __int64)v9 > 0x80 )
  {
    v56 = v11;
    sub_C8D5F0((__int64)&v68, v70, v9 >> 5, 8u, v11, a6);
    v13 = v69;
    v11 = v56;
    v12 = &v68[8 * (unsigned int)v69];
  }
  if ( v10 != (_QWORD *)v11 )
  {
    do
    {
      if ( v12 )
        *v12 = *v10;
      v10 += 4;
      ++v12;
    }
    while ( v10 != (_QWORD *)v11 );
    v13 = v69;
  }
  v62[0] = 0;
  LODWORD(v69) = v14 + v13;
  v15 = a2;
  v63 = a2;
  v62[1] = 0;
  v16 = a2 != -8192 && a2 != -4096;
  if ( v16 )
  {
    sub_BD73F0((__int64)v62);
    v15 = v63;
  }
  v17 = *(_DWORD *)(a1 + 56);
  if ( v17 )
  {
    v18 = v17 - 1;
    v19 = *(_QWORD *)(a1 + 40);
    v64 = 0;
    v65 = 0;
    v20 = (v17 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v66 = -4096;
    v21 = (_QWORD *)(v19 + 32LL * v20);
    v22 = v21[2];
    if ( v15 == v22 )
    {
LABEL_14:
      v54 = v21;
      sub_D68D70(&v64);
      v23 = v54;
      v64 = 0;
      v65 = 0;
      v66 = -8192;
      v24 = v54[2];
      if ( v24 != -8192 )
      {
        if ( v24 != -4096 && v24 )
        {
          sub_BD60C0(v54);
          v23 = v54;
        }
        v23[2] = -8192;
      }
      sub_D68D70(&v64);
      --*(_DWORD *)(a1 + 48);
      v15 = v63;
      ++*(_DWORD *)(a1 + 52);
    }
    else
    {
      v50 = 1;
      while ( v22 != -4096 )
      {
        v53 = v50 + 1;
        v20 = v18 & (v50 + v20);
        v21 = (_QWORD *)(v19 + 32LL * v20);
        v22 = v21[2];
        if ( v15 == v22 )
          goto LABEL_14;
        v50 = v53;
      }
      sub_D68D70(&v64);
      v15 = v63;
    }
  }
  if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
    sub_BD60C0(v62);
  v57 = 0;
  v25 = a2;
  v58 = 0;
  v59 = a2;
  if ( v16 )
  {
    sub_BD73F0((__int64)&v57);
    v25 = v59;
  }
  v26 = *(_DWORD *)(a3 + 24);
  if ( v26 )
  {
    v27 = v26 - 1;
    v28 = *(_QWORD *)(a3 + 8);
    v64 = 0;
    v65 = 0;
    v29 = (v26 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v66 = -4096;
    v30 = (_QWORD *)(v28 + 24LL * v29);
    v31 = v30[2];
    if ( v25 == v31 )
    {
LABEL_27:
      v55 = v30;
      sub_D68D70(&v64);
      v32 = v55;
      v64 = 0;
      v65 = 0;
      v66 = -8192;
      v33 = v55[2];
      if ( v33 != -8192 )
      {
        if ( v33 != -4096 && v33 )
        {
          sub_BD60C0(v55);
          v32 = v55;
        }
        v32[2] = -8192;
      }
      sub_D68D70(&v64);
      --*(_DWORD *)(a3 + 16);
      ++*(_DWORD *)(a3 + 20);
      sub_28ED760(v60, (_QWORD *)(a3 + 32), (__int64)&v57);
      v64 = v60[0];
      v34 = *v61;
      v67 = v61;
      v65 = v34;
      v66 = v34 + 504;
      sub_28FB170(v62, (_QWORD *)(a3 + 32), (__int64)&v64);
      v25 = v59;
    }
    else
    {
      v51 = 1;
      while ( v31 != -4096 )
      {
        v52 = v51 + 1;
        v29 = v27 & (v51 + v29);
        v30 = (_QWORD *)(v28 + 24LL * v29);
        v31 = v30[2];
        if ( v25 == v31 )
          goto LABEL_27;
        v51 = v52;
      }
      sub_D68D70(&v64);
      v25 = v59;
    }
  }
  if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
    sub_BD60C0(&v57);
  v57 = 0;
  v35 = a2;
  v58 = 0;
  v59 = a2;
  if ( v16 )
  {
    sub_BD73F0((__int64)&v57);
    v35 = v59;
  }
  v36 = *(_DWORD *)(a1 + 88);
  if ( v36 )
  {
    v37 = v36 - 1;
    v38 = *(_QWORD *)(a1 + 72);
    v64 = 0;
    v65 = 0;
    v39 = (v36 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v66 = -4096;
    v40 = (_QWORD *)(v38 + 24LL * v39);
    v41 = v40[2];
    if ( v35 == v41 )
    {
LABEL_40:
      sub_D68D70(&v64);
      v64 = 0;
      v65 = 0;
      v66 = -8192;
      v42 = v40[2];
      if ( v42 != -8192 )
      {
        if ( v42 && v42 != -4096 )
          sub_BD60C0(v40);
        v40[2] = -8192;
      }
      sub_D68D70(&v64);
      --*(_DWORD *)(a1 + 80);
      ++*(_DWORD *)(a1 + 84);
      v43 = (_QWORD *)(a1 + 96);
      sub_28ED760(v60, v43, (__int64)&v57);
      v64 = v60[0];
      v44 = *v61;
      v67 = v61;
      v65 = v44;
      v66 = v44 + 504;
      sub_28FB170(v62, v43, (__int64)&v64);
      v35 = v59;
    }
    else
    {
      v49 = 1;
      while ( v41 != -4096 )
      {
        v39 = v37 & (v49 + v39);
        v40 = (_QWORD *)(v38 + 24LL * v39);
        v41 = v40[2];
        if ( v41 == v35 )
          goto LABEL_40;
        ++v49;
      }
      sub_D68D70(&v64);
      v35 = v59;
    }
  }
  if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
    sub_BD60C0(&v57);
  sub_F54ED0((unsigned __int8 *)a2);
  sub_B43D60((_QWORD *)a2);
  v45 = v68;
  v46 = &v68[8 * (unsigned int)v69];
  if ( v46 != (_QWORD *)v68 )
  {
    v47 = v68;
    do
    {
      while ( 1 )
      {
        v48 = (_BYTE *)*v47;
        if ( *(_BYTE *)*v47 > 0x1Cu && !*((_QWORD *)v48 + 2) )
        {
          v64 = 0;
          v65 = 0;
          v66 = (__int64)v48;
          if ( v48 != (_BYTE *)-8192LL && v48 != (_BYTE *)-4096LL )
            sub_BD73F0((__int64)&v64);
          sub_28F19A0(a3, &v64);
          if ( v66 != -4096 && v66 != 0 && v66 != -8192 )
            break;
        }
        if ( v46 == ++v47 )
          goto LABEL_60;
      }
      ++v47;
      sub_BD60C0(&v64);
    }
    while ( v46 != v47 );
LABEL_60:
    v45 = v68;
  }
  if ( v45 != v70 )
    _libc_free((unsigned __int64)v45);
}
