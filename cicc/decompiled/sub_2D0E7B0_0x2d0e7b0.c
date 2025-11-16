// Function: sub_2D0E7B0
// Address: 0x2d0e7b0
//
__int64 __fastcall sub_2D0E7B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // [rsp+8h] [rbp-168h]
  __int64 v24; // [rsp+10h] [rbp-160h] BYREF
  _QWORD *v25; // [rsp+18h] [rbp-158h]
  __int64 v26; // [rsp+20h] [rbp-150h]
  __int64 v27; // [rsp+28h] [rbp-148h]
  _QWORD v28[2]; // [rsp+30h] [rbp-140h] BYREF
  __int64 v29; // [rsp+40h] [rbp-130h] BYREF
  _QWORD *v30; // [rsp+48h] [rbp-128h]
  __int64 v31; // [rsp+50h] [rbp-120h]
  __int64 v32; // [rsp+58h] [rbp-118h]
  _QWORD v33[2]; // [rsp+60h] [rbp-110h] BYREF
  int v34; // [rsp+70h] [rbp-100h]
  _QWORD v35[3]; // [rsp+78h] [rbp-F8h] BYREF
  _QWORD v36[6]; // [rsp+90h] [rbp-E0h] BYREF
  int v37; // [rsp+C0h] [rbp-B0h]
  char v38; // [rsp+C4h] [rbp-ACh]
  char v39; // [rsp+C8h] [rbp-A8h] BYREF
  __int64 v40; // [rsp+108h] [rbp-68h]
  __int64 v41; // [rsp+110h] [rbp-60h]
  __int64 v42; // [rsp+118h] [rbp-58h]
  int v43; // [rsp+120h] [rbp-50h]
  __int64 v44; // [rsp+128h] [rbp-48h]
  __int64 v45; // [rsp+130h] [rbp-40h]
  __int64 v46; // [rsp+138h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v23 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = sub_BC1CD0(a4, &unk_4F8D9A8, a3);
  v8 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v9 = *(_QWORD *)(a3 + 40);
  v24 = a3;
  v26 = v6 + 8;
  v28[1] = v8 + 8;
  WORD2(v30) = 256;
  v35[1] = v35;
  v35[0] = v35;
  v36[1] = v36;
  v36[0] = v36;
  v25 = (_QWORD *)(v9 + 312);
  v27 = v23 + 8;
  v36[4] = &v39;
  v28[0] = v7 + 8;
  v29 = 0;
  LODWORD(v30) = 5;
  BYTE6(v30) = 0;
  LODWORD(v31) = 30;
  v32 = 0;
  v33[0] = 0;
  v33[1] = 0;
  v34 = 0;
  v35[2] = 0;
  v36[2] = 0;
  v36[3] = 0;
  v36[5] = 8;
  v37 = 0;
  v38 = 1;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  LOBYTE(v23) = sub_2D0DB80((__int64)&v24);
  sub_2D05B80((__int64)&v24);
  if ( (_BYTE)v23 )
  {
    v30 = v33;
    v25 = v28;
    v24 = 0;
    v26 = 2;
    LODWORD(v27) = 0;
    BYTE4(v27) = 1;
    v29 = 0;
    v31 = 2;
    LODWORD(v32) = 0;
    BYTE4(v32) = 1;
    sub_AE6EC0((__int64)&v24, (__int64)&unk_4F875F0);
    sub_2D04AA0((__int64)&v24, (__int64)&unk_4F81450, v11, v12, v13, v14);
    sub_2D04AA0((__int64)&v24, (__int64)&unk_4F8D9A8, v15, v16, v17, v18);
    sub_2D04AA0((__int64)&v24, (__int64)&unk_4F89C30, v19, v20, v21, v22);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v28, (__int64)&v24);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v33, (__int64)&v29);
    if ( BYTE4(v32) )
    {
      if ( BYTE4(v27) )
        return a1;
    }
    else
    {
      _libc_free((unsigned __int64)v30);
      if ( BYTE4(v27) )
        return a1;
    }
    _libc_free((unsigned __int64)v25);
    return a1;
  }
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&qword_4F82400);
  return a1;
}
