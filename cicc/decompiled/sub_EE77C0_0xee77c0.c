// Function: sub_EE77C0
// Address: 0xee77c0
//
__int64 __fastcall sub_EE77C0(__int64 a1, __int64 *a2, __int64 *a3, unsigned __int8 *a4, int *a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 *v24; // r13
  unsigned __int64 *v25; // r15
  unsigned __int64 *v26; // r12
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  _BYTE **v34; // rsi
  _QWORD *v35; // rax
  __int64 v36; // r15
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v40; // rax
  _BYTE *v41; // rdx
  _BYTE *v42; // rdi
  unsigned __int8 v43; // r9
  __int64 *v44; // rdx
  unsigned __int64 *v49; // [rsp+28h] [rbp-E8h]
  char v50; // [rsp+36h] [rbp-DAh]
  unsigned __int8 v51; // [rsp+37h] [rbp-D9h]
  int v52; // [rsp+38h] [rbp-D8h]
  __int64 *v53; // [rsp+48h] [rbp-C8h] BYREF
  _BYTE *v54; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+58h] [rbp-B8h]
  _BYTE v56[176]; // [rsp+60h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = *a2;
  v50 = *(_BYTE *)(a1 + 129);
  v55 = 0x2000000000LL;
  v52 = *a5;
  v51 = *a4;
  v9 = *a3;
  v10 = a3[1];
  v54 = v56;
  v49 = (unsigned __int64 *)v9;
  sub_D953B0((__int64)&v54, 63, v9, (__int64)a4, (__int64)a5, a6);
  sub_D953B0((__int64)&v54, v7, v11, v12, v13, v14);
  sub_D953B0((__int64)&v54, v10, v15, v16, v17, v18);
  v22 = (__int64)v49;
  v23 = (unsigned int)v55;
  v24 = &v49[v10];
  if ( v49 != v24 )
  {
    v25 = v49;
    v26 = v24;
    do
    {
      v27 = *v25;
      if ( v23 + 1 > (unsigned __int64)HIDWORD(v55) )
      {
        sub_C8D5F0((__int64)&v54, v56, v23 + 1, 4u, v20, v21);
        v23 = (unsigned int)v55;
      }
      *(_DWORD *)&v54[4 * v23] = v27;
      v28 = HIDWORD(v27);
      v19 = HIDWORD(v55);
      LODWORD(v55) = v55 + 1;
      v29 = (unsigned int)v55;
      if ( (unsigned __int64)(unsigned int)v55 + 1 > HIDWORD(v55) )
      {
        sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 4u, v20, v21);
        v29 = (unsigned int)v55;
      }
      v22 = (__int64)v54;
      ++v25;
      *(_DWORD *)&v54[4 * v29] = v28;
      v23 = (unsigned int)(v55 + 1);
      LODWORD(v55) = v55 + 1;
    }
    while ( v26 != v25 );
    v6 = a1;
  }
  sub_D953B0((__int64)&v54, v51, v22, v19, v20, v21);
  sub_D953B0((__int64)&v54, v52, v30, v31, v32, v33);
  v34 = &v54;
  v35 = sub_C65B40(v6 + 96, (__int64)&v54, (__int64 *)&v53, (__int64)off_497B2F0);
  v36 = (__int64)v35;
  if ( v35 )
  {
    v36 = (__int64)(v35 + 1);
    if ( v54 != v56 )
      _libc_free(v54, &v54);
    v54 = (_BYTE *)v36;
    v37 = sub_EE6840(v6 + 136, (__int64 *)&v54);
    if ( v37 )
    {
      v38 = v37[1];
      if ( v38 )
        v36 = v38;
    }
    if ( *(_QWORD *)(v6 + 120) == v36 )
      *(_BYTE *)(v6 + 128) = 1;
  }
  else
  {
    if ( v50 )
    {
      v40 = sub_CD1D40((__int64 *)v6, 56, 3);
      *(_QWORD *)v40 = 0;
      v34 = (_BYTE **)v40;
      v36 = v40 + 8;
      v41 = (_BYTE *)*a3;
      v42 = (_BYTE *)a3[1];
      v43 = *a4;
      LODWORD(v40) = *a5;
      v34[3] = (_BYTE *)*a2;
      v34[4] = v41;
      v44 = v53;
      v34[5] = v42;
      *((_BYTE *)v34 + 48) = v43;
      *((_WORD *)v34 + 8) = ((v40 & 0x3F) << 8) | 0x403F;
      *((_BYTE *)v34 + 18) = *((_BYTE *)v34 + 18) & 0xF0 | 5;
      v34[1] = &unk_49E0568;
      sub_C657C0((__int64 *)(v6 + 96), (__int64 *)v34, v44, (__int64)off_497B2F0);
    }
    if ( v54 != v56 )
      _libc_free(v54, v34);
    *(_QWORD *)(v6 + 112) = v36;
  }
  return v36;
}
