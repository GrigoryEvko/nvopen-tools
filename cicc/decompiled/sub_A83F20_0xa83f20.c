// Function: sub_A83F20
// Address: 0xa83f20
//
__int64 __fastcall sub_A83F20(unsigned __int8 ***a1)
{
  unsigned __int8 *v2; // rbx
  __int64 v3; // rdi
  int v4; // edx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // eax
  unsigned __int8 *v14; // r13
  __int64 v15; // rax
  unsigned __int8 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r14
  int v19; // ecx
  int v20; // r8d
  _QWORD *v21; // rax
  __int64 *v22; // rax
  unsigned __int64 v23; // rsi
  unsigned int **v24; // rdi
  int v25; // r8d
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // r8d
  __int64 v29; // rbx
  __int64 v30; // rax
  _BYTE *v31; // r10
  unsigned int **v32; // rdi
  __int64 v33; // rax
  unsigned int **v34; // r15
  unsigned int *v35; // rdi
  __int64 (__fastcall *v36)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // r9
  __int64 v40; // rbx
  unsigned int *v41; // r15
  unsigned int *v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 result; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // [rsp+8h] [rbp-108h]
  _BYTE *v52; // [rsp+10h] [rbp-100h]
  _BYTE *v53; // [rsp+10h] [rbp-100h]
  _BYTE *v54; // [rsp+10h] [rbp-100h]
  __int64 v55; // [rsp+18h] [rbp-F8h]
  __int64 v56; // [rsp+20h] [rbp-F0h]
  unsigned int v57; // [rsp+3Ch] [rbp-D4h] BYREF
  char v58[32]; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v59; // [rsp+60h] [rbp-B0h]
  _BYTE v60[32]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v61; // [rsp+90h] [rbp-80h]
  _BYTE *v62; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v63; // [rsp+A8h] [rbp-68h]
  _BYTE v64[96]; // [rsp+B0h] [rbp-60h] BYREF

  v2 = **a1;
  v3 = (__int64)*a1[1];
  if ( *((_QWORD *)v2 + 10) == *(_QWORD *)(v3 + 24) )
  {
    if ( *((_QWORD *)v2 - 4) )
    {
      v50 = *((_QWORD *)v2 - 3);
      **((_QWORD **)v2 - 2) = v50;
      if ( v50 )
        *(_QWORD *)(v50 + 16) = *((_QWORD *)v2 - 2);
    }
    *((_QWORD *)v2 - 4) = v3;
    result = *(_QWORD *)(v3 + 16);
    *((_QWORD *)v2 - 3) = result;
    if ( result )
      *(_QWORD *)(result + 16) = v2 - 24;
    *((_QWORD *)v2 - 2) = v3 + 16;
    *(_QWORD *)(v3 + 16) = v2 - 32;
    return result;
  }
  v56 = *((_QWORD *)v2 + 1);
  if ( *(_BYTE *)(v56 + 8) != 15 )
  {
    result = sub_ADAFB0(v3, *(_QWORD *)(*((_QWORD *)v2 - 4) + 8LL));
    if ( *((_QWORD *)v2 - 4) )
    {
      v48 = *((_QWORD *)v2 - 3);
      **((_QWORD **)v2 - 2) = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = *((_QWORD *)v2 - 2);
    }
    *((_QWORD *)v2 - 4) = result;
    if ( result )
    {
      v49 = *(_QWORD *)(result + 16);
      *((_QWORD *)v2 - 3) = v49;
      if ( v49 )
        *(_QWORD *)(v49 + 16) = v2 - 24;
      *((_QWORD *)v2 - 2) = result + 16;
      *(_QWORD *)(result + 16) = v2 - 32;
    }
    return result;
  }
  v4 = *v2;
  if ( v4 != 40 )
  {
    v5 = -32;
    if ( v4 != 85 )
    {
      v5 = -96;
      if ( v4 != 34 )
        BUG();
    }
    if ( (v2[7] & 0x80u) == 0 )
      goto LABEL_12;
    goto LABEL_6;
  }
  v5 = -32 - 32LL * (unsigned int)sub_B491D0(v2);
  if ( (v2[7] & 0x80u) != 0 )
  {
LABEL_6:
    v6 = sub_BD2BC0(v2);
    v8 = v6 + v7;
    v9 = 0;
    if ( (v2[7] & 0x80u) != 0 )
      v9 = sub_BD2BC0(v2);
    if ( (unsigned int)((v8 - v9) >> 4) )
    {
      if ( (v2[7] & 0x80u) == 0 )
        BUG();
      v10 = *(_DWORD *)(sub_BD2BC0(v2) + 8);
      if ( (v2[7] & 0x80u) == 0 )
        BUG();
      v11 = sub_BD2BC0(v2);
      v5 -= 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
    }
  }
LABEL_12:
  v13 = *((_DWORD *)v2 + 1);
  v14 = &v2[v5];
  v62 = v64;
  v15 = 32LL * (v13 & 0x7FFFFFF);
  v63 = 0x600000000LL;
  v16 = &v2[-v15];
  v17 = v5 + v15;
  v18 = v17 >> 5;
  if ( (unsigned __int64)v17 > 0xC0 )
  {
    sub_C8D5F0(&v62, v64, v17 >> 5, 8);
    v19 = (int)v62;
    v20 = v63;
    v21 = &v62[8 * (unsigned int)v63];
  }
  else
  {
    v19 = (unsigned int)v64;
    v20 = 0;
    v21 = v64;
  }
  if ( v16 != v14 )
  {
    do
    {
      if ( v21 )
        *v21 = *(_QWORD *)v16;
      v16 += 32;
      ++v21;
    }
    while ( v14 != v16 );
    v19 = (int)v62;
    v20 = v63;
  }
  v22 = (__int64 *)a1[1];
  v23 = 0;
  LODWORD(v63) = v18 + v20;
  v24 = (unsigned int **)a1[2];
  v25 = v18 + v20;
  v61 = 257;
  v26 = *v22;
  if ( *v22 )
    v23 = *(_QWORD *)(v26 + 24);
  v55 = sub_921880(v24, v23, v26, v19, v25, (__int64)v60, 0);
  *(_QWORD *)(v55 + 72) = *((_QWORD *)**a1 + 9);
  v27 = sub_ACADE0(v56);
  v28 = *(_DWORD *)(v56 + 12);
  v57 = 0;
  v29 = v27;
  if ( v28 )
  {
    do
    {
      v32 = (unsigned int **)a1[2];
      v61 = 257;
      v33 = sub_94D3D0(v32, v55, (__int64)&v57, 1, (__int64)v60);
      v34 = (unsigned int **)a1[2];
      v59 = 257;
      v31 = (_BYTE *)v33;
      v35 = v34[10];
      v36 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v35 + 88LL);
      if ( v36 == sub_9482E0 )
      {
        if ( *(_BYTE *)v29 > 0x15u || *v31 > 0x15u )
        {
LABEL_30:
          v53 = v31;
          v61 = 257;
          v37 = sub_BD2C40(104, unk_3F148BC);
          v38 = v37;
          if ( v37 )
          {
            v39 = v51;
            LOWORD(v39) = 0;
            sub_B44260(v37, *(_QWORD *)(v29 + 8), 65, 2, 0, v39);
            *(_QWORD *)(v38 + 72) = v38 + 88;
            *(_QWORD *)(v38 + 80) = 0x400000000LL;
            sub_B4FD20(v38, v29, v53, &v57, 1, v60);
          }
          (*(void (__fastcall **)(unsigned int *, __int64, char *, unsigned int *, unsigned int *))(*(_QWORD *)v34[11]
                                                                                                  + 16LL))(
            v34[11],
            v38,
            v58,
            v34[7],
            v34[8]);
          v40 = 4LL * *((unsigned int *)v34 + 2);
          v41 = *v34;
          v42 = &v41[v40];
          while ( v42 != v41 )
          {
            v43 = *((_QWORD *)v41 + 1);
            v44 = *v41;
            v41 += 4;
            sub_B99FD0(v38, v44, v43);
          }
          v29 = v38;
          goto LABEL_27;
        }
        v52 = v31;
        v30 = sub_AAAE30(v29, v31, &v57, 1);
        v31 = v52;
      }
      else
      {
        v54 = v31;
        v30 = v36((__int64)v35, (_BYTE *)v29, v31, (__int64)&v57, 1);
        v31 = v54;
      }
      if ( !v30 )
        goto LABEL_30;
      v29 = v30;
LABEL_27:
      ++v57;
    }
    while ( v57 < *(_DWORD *)(v56 + 12) );
  }
  sub_BD84D0(**a1, v29);
  result = sub_B43D60(**a1, v29, v45, v46);
  if ( v62 != v64 )
    return _libc_free(v62, v29);
  return result;
}
