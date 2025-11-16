// Function: sub_321AA10
// Address: 0x321aa10
//
void __fastcall sub_321AA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 i; // r12
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r14
  _BYTE *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 v22; // r15
  __int64 j; // r14
  __int64 v24; // r13
  __int64 v25; // r12
  __int64 v26; // rdx
  char *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r13
  __int64 v30; // r12
  __int64 v31; // rdx
  char v32; // dl
  __int64 v35; // [rsp+18h] [rbp-E8h]
  __int64 v36; // [rsp+28h] [rbp-D8h]
  char *v37; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v38; // [rsp+30h] [rbp-D0h]
  char v39[8]; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v40; // [rsp+48h] [rbp-B8h]
  char v41[8]; // [rsp+60h] [rbp-A0h] BYREF
  char *v42; // [rsp+68h] [rbp-98h]
  __int64 v43; // [rsp+80h] [rbp-80h] BYREF
  char *v44[2]; // [rsp+88h] [rbp-78h] BYREF
  _BYTE v45[48]; // [rsp+98h] [rbp-68h] BYREF
  char v46; // [rsp+C8h] [rbp-38h]

  v6 = a1 + 80 * a2;
  v7 = v6 + 8;
  v35 = (a3 - 1) / 2;
  if ( a2 >= v35 )
  {
    v16 = a2;
    v17 = v6 + 8;
  }
  else
  {
    for ( i = a2; ; i = v9 )
    {
      v9 = 2 * (i + 1);
      v10 = 160 * (i + 1);
      v11 = a1 + v10 - 80;
      v6 = a1 + v10;
      v36 = v11;
      sub_AF47B0(
        (__int64)v41,
        *(unsigned __int64 **)(*(_QWORD *)v6 + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)v6 + 24LL));
      v37 = v42;
      sub_AF47B0(
        (__int64)&v43,
        *(unsigned __int64 **)(*(_QWORD *)v36 + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)v36 + 24LL));
      if ( v37 < v44[0] )
      {
        --v9;
        v6 = a1 + 80 * v9;
      }
      v15 = *(_QWORD *)v6;
      *(_QWORD *)(a1 + 80 * i) = *(_QWORD *)v6;
      sub_3218940(v7, (char **)(v6 + 8), v15, v12, v13, v14);
      *(_BYTE *)(a1 + 80 * i + 72) = *(_BYTE *)(v6 + 72);
      if ( v9 >= v35 )
        break;
      v7 = v6 + 8;
    }
    v16 = v9;
    v17 = v6 + 8;
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v16 )
  {
    v28 = v16 + 1;
    v29 = 2 * (v16 + 1);
    v30 = a1 + 160 * (v16 + 1) - 80;
    v31 = *(_QWORD *)v30;
    *(_QWORD *)v6 = *(_QWORD *)v30;
    sub_3218940(v17, (char **)(v30 + 8), v31, v28, a5, a6);
    v32 = *(_BYTE *)(v30 + 72);
    v16 = v29 - 1;
    *(_BYTE *)(v6 + 72) = v32;
    v6 = a1 + 80 * (v29 - 1);
    v17 = v6 + 8;
  }
  v18 = v45;
  v44[1] = (char *)0x200000000LL;
  v19 = *(_QWORD *)a4;
  v44[0] = v45;
  v43 = v19;
  v20 = *(unsigned int *)(a4 + 16);
  if ( (_DWORD)v20 )
    sub_3218940((__int64)v44, (char **)(a4 + 8), v20, (__int64)v45, a5, a6);
  v46 = *(_BYTE *)(a4 + 72);
  v21 = (v16 - 1) / 2;
  if ( v16 > a2 )
  {
    v22 = v17;
    for ( j = (v16 - 1) / 2; ; j = (j - 1) / 2 )
    {
      v24 = a1 + 80 * j;
      sub_AF47B0(
        (__int64)v39,
        *(unsigned __int64 **)(*(_QWORD *)v24 + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)v24 + 24LL));
      v38 = v40;
      sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v43 + 16), *(unsigned __int64 **)(v43 + 24));
      v18 = (_BYTE *)v38;
      v25 = a1 + 80 * v16;
      if ( v38 >= (unsigned __int64)v42 )
      {
        v17 = v22;
        v6 = v25;
        goto LABEL_19;
      }
      *(_QWORD *)v25 = *(_QWORD *)v24;
      sub_3218940(v22, (char **)(v24 + 8), v24 + 8, v38, a5, v21);
      *(_BYTE *)(v25 + 72) = *(_BYTE *)(v24 + 72);
      v18 = (_BYTE *)(j - 1);
      v16 = j;
      if ( j <= a2 )
        break;
      v22 = v24 + 8;
    }
    v17 = v24 + 8;
    v6 = v24;
  }
LABEL_19:
  v26 = v43;
  *(_QWORD *)v6 = v43;
  sub_3218940(v17, v44, v26, (__int64)v18, a5, v21);
  v27 = v44[0];
  *(_BYTE *)(v6 + 72) = v46;
  if ( v27 != v45 )
    _libc_free((unsigned __int64)v27);
}
