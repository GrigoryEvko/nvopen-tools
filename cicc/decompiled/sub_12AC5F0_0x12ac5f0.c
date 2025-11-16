// Function: sub_12AC5F0
// Address: 0x12ac5f0
//
__int64 __fastcall sub_12AC5F0(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int v6; // r9d
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r13
  char *v10; // r12
  char *v11; // r15
  char *v12; // rax
  _QWORD *v13; // rdi
  char *v14; // r13
  __int64 v15; // rax
  unsigned int v16; // r12d
  __int64 v17; // r14
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // rdi
  unsigned __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rsi
  _QWORD *v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rsi
  _QWORD *v33; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // [rsp+8h] [rbp-138h]
  int v40; // [rsp+2Ch] [rbp-114h]
  char *v41; // [rsp+30h] [rbp-110h]
  unsigned int v42; // [rsp+38h] [rbp-108h]
  __int64 *v43; // [rsp+38h] [rbp-108h]
  __int64 *v44; // [rsp+40h] [rbp-100h]
  unsigned int v45; // [rsp+48h] [rbp-F8h]
  __int64 v46; // [rsp+48h] [rbp-F8h]
  unsigned __int64 *v47; // [rsp+48h] [rbp-F8h]
  __int64 v48; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-E8h] BYREF
  _BYTE v50[16]; // [rsp+60h] [rbp-E0h] BYREF
  __int16 v51; // [rsp+70h] [rbp-D0h]
  _QWORD *v52; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+88h] [rbp-B8h]
  _QWORD v54[22]; // [rsp+90h] [rbp-B0h] BYREF

  v4 = (unsigned int)(a3 - 678);
  if ( (unsigned int)v4 > 0x1D )
  {
    v35 = (unsigned int)(a3 - 708);
    if ( (unsigned int)v35 > 0x17 )
    {
      v37 = (unsigned int)(a3 - 732);
      if ( (unsigned int)v37 > 0xC )
        sub_127B630("unexpected WMMA intrinsic!", 0);
      v45 = dword_4281020[v37];
      v6 = v45 - 3971;
    }
    else
    {
      v45 = dword_4281060[v35];
      v6 = v45 - 3971;
    }
  }
  else
  {
    v45 = dword_42810C0[v4];
    v6 = v45 - 3971;
  }
  v42 = v6;
  v7 = *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 16LL);
  v44 = *(__int64 **)(v7 + 16);
  v9 = *(_QWORD *)(v8 + 16);
  sub_12A6F10(v9, 1u, "unexpected 'rowcol' operand", "'rowcol' operand can be 0 or 1 only", (_DWORD *)(a4 + 36));
  v10 = sub_128F980((__int64)a2, v7);
  v41 = sub_128F980((__int64)a2, (__int64)v44);
  v11 = sub_128F980((__int64)a2, v8);
  v12 = sub_128F980((__int64)a2, v9);
  v13 = (_QWORD *)a2[4];
  v14 = v12;
  v48 = *(_QWORD *)v10;
  v15 = sub_126A190(v13, v45, (__int64)&v48, 1u);
  v54[0] = v10;
  v38 = v15;
  v52 = v54;
  v54[1] = v11;
  v54[2] = v14;
  v53 = 0x1000000003LL;
  if ( v42 > 0x3A || (v36 = 0x404040001004001LL, v40 = 8, !_bittest64(&v36, v42)) )
  {
    if ( v45 == 4037 || (v40 = 4, v45 == 3753) )
      v40 = 2;
  }
  v16 = 0;
  v43 = a2 + 6;
  do
  {
    v17 = a2[4] + 8LL;
    v18 = sub_8D46C0(*v44);
    v19 = sub_127A030(v17, v18, 0);
    v51 = 257;
    v20 = v19;
    v21 = sub_12A8800(v43, v19, v41, v16, (__int64)v50);
    v51 = 257;
    v46 = v21;
    v22 = sub_1648A60(64, 1);
    v23 = (_QWORD *)v22;
    if ( v22 )
      sub_15F9210(v22, v20, v46, 0, 0, 0);
    v24 = a2[7];
    if ( v24 )
    {
      v47 = (unsigned __int64 *)a2[8];
      sub_157E9D0(v24 + 40, v23);
      v25 = *v47;
      v26 = v23[3] & 7LL;
      v23[4] = v47;
      v25 &= 0xFFFFFFFFFFFFFFF8LL;
      v23[3] = v25 | v26;
      *(_QWORD *)(v25 + 8) = v23 + 3;
      *v47 = *v47 & 7 | (unsigned __int64)(v23 + 3);
    }
    sub_164B780(v23, v50);
    v27 = a2[6];
    if ( v27 )
    {
      v49 = a2[6];
      sub_1623A60(&v49, v27, 2);
      v28 = v23 + 6;
      if ( v23[6] )
      {
        sub_161E7C0(v23 + 6);
        v28 = v23 + 6;
      }
      v29 = v49;
      v23[6] = v49;
      if ( v29 )
        sub_1623210(&v49, v29, v28);
    }
    v30 = (unsigned int)v53;
    if ( (unsigned int)v53 >= HIDWORD(v53) )
    {
      sub_16CD150(&v52, v54, 0, 8);
      v30 = (unsigned int)v53;
    }
    ++v16;
    v52[v30] = v23;
    v31 = (unsigned int)(v53 + 1);
    LODWORD(v53) = v53 + 1;
  }
  while ( v40 != v16 );
  v51 = 257;
  v32 = *(_QWORD *)(v38 + 24);
  sub_1285290(v43, v32, v38, (int)v52, v31, (__int64)v50, 0);
  v33 = v52;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v33 != v54 )
    _libc_free(v33, v32);
  return a1;
}
