// Function: sub_30315D0
// Address: 0x30315d0
//
__int64 __fastcall sub_30315D0(__int64 a1, int a2, int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int128 v12; // rax
  int v13; // r9d
  __int128 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rcx
  int v27; // r8d
  __int64 v28; // r13
  __int128 v30; // [rsp-10h] [rbp-D0h]
  __int64 v31; // [rsp+0h] [rbp-C0h]
  __int64 v32; // [rsp+8h] [rbp-B8h]
  int v33; // [rsp+18h] [rbp-A8h]
  unsigned int v34; // [rsp+20h] [rbp-A0h]
  __int64 v35; // [rsp+28h] [rbp-98h]
  __int64 v36; // [rsp+30h] [rbp-90h] BYREF
  int v37; // [rsp+38h] [rbp-88h]
  _BYTE *v38; // [rsp+40h] [rbp-80h] BYREF
  __int64 v39; // [rsp+48h] [rbp-78h]
  _BYTE v40[112]; // [rsp+50h] [rbp-70h] BYREF

  v8 = *(_QWORD *)(a1 + 80);
  v36 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v36, v8, 1);
  v9 = *(_QWORD *)(a1 + 40);
  v37 = *(_DWORD *)(a1 + 72);
  v10 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 96LL);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v33 = (int)v11;
  if ( (BYTE1(v11) & 1) != 0 )
  {
    *(_QWORD *)&v12 = sub_3400BD0(a2, 16, (unsigned int)&v36, 7, 0, 0, 0);
    *(_QWORD *)&v14 = sub_3406EB0(
                        a2,
                        192,
                        (unsigned int)&v36,
                        7,
                        0,
                        v13,
                        *(_OWORD *)(*(_QWORD *)(a1 + 40) + 120LL),
                        v12);
    v30 = v14;
  }
  else
  {
    v30 = *(_OWORD *)(v9 + 120);
  }
  v15 = sub_33FAF80(a2, 216, (unsigned int)&v36, 6, 0, a6, v30);
  v34 = v16;
  v35 = v15;
  v38 = v40;
  v39 = 0x400000000LL;
  v17 = sub_3400BD0(a2, v33, (unsigned int)&v36, 8, 0, 1, 0);
  v18 = (unsigned int)v39;
  v20 = v19;
  v21 = (unsigned int)v39 + 1LL;
  if ( v21 > HIDWORD(v39) )
  {
    v31 = v17;
    v32 = v20;
    sub_C8D5F0((__int64)&v38, v40, v21, 0x10u, v17, v20);
    v18 = (unsigned int)v39;
    v17 = v31;
    v20 = v32;
  }
  v22 = (__int64 *)&v38[16 * v18];
  *v22 = v17;
  v23 = v34;
  v22[1] = v20;
  LODWORD(v39) = v39 + 1;
  v24 = (unsigned int)v39;
  if ( (unsigned __int64)(unsigned int)v39 + 1 > HIDWORD(v39) )
  {
    sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 0x10u, v34, v20);
    v24 = (unsigned int)v39;
    v23 = v34;
  }
  v25 = (__int64 *)&v38[16 * v24];
  v25[1] = v23;
  *v25 = v35;
  v26 = *(_QWORD *)(a1 + 48);
  v27 = *(_DWORD *)(a1 + 68);
  LODWORD(v39) = v39 + 1;
  v28 = sub_33E66D0(a2, a3, (unsigned int)&v36, v26, v27, v20, (__int64)v38, (unsigned int)v39);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v28;
}
