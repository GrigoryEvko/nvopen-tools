// Function: sub_2CB19C0
// Address: 0x2cb19c0
//
__int64 __fastcall sub_2CB19C0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r10
  __int64 *v9; // r13
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // r10
  unsigned __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rdx
  unsigned __int64 v22; // r8
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // r15
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // r13
  int v33; // ecx
  bool v34; // zf
  __m128i v35; // xmm0
  __int64 v37; // [rsp+0h] [rbp-110h]
  __int64 v38; // [rsp+0h] [rbp-110h]
  int v39; // [rsp+8h] [rbp-108h]
  __int64 v40; // [rsp+10h] [rbp-100h]
  __int64 v41; // [rsp+10h] [rbp-100h]
  __int64 v42; // [rsp+10h] [rbp-100h]
  __int64 v43; // [rsp+10h] [rbp-100h]
  _QWORD *v44; // [rsp+18h] [rbp-F8h]
  __int64 *v45; // [rsp+18h] [rbp-F8h]
  __m128i v48; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v49; // [rsp+40h] [rbp-D0h]
  _BYTE *v50; // [rsp+58h] [rbp-B8h] BYREF
  const char *v51; // [rsp+60h] [rbp-B0h] BYREF
  char v52; // [rsp+80h] [rbp-90h]
  char v53; // [rsp+81h] [rbp-8Fh]
  __int64 *v54; // [rsp+90h] [rbp-80h] BYREF
  __int64 v55; // [rsp+98h] [rbp-78h]
  _QWORD v56[14]; // [rsp+A0h] [rbp-70h] BYREF

  v7 = a2;
  v9 = *(__int64 **)(a5 + 40);
  v10 = *(_QWORD *)(a6 + 104);
  v11 = (_QWORD *)*v9;
  v50 = (_BYTE *)v10;
  v44 = v11;
  v12 = *(_QWORD *)(a7 + 16);
  if ( v12 )
  {
    v56[0] = v10;
    v56[1] = v12;
    v54 = v56;
    v55 = 0x200000002LL;
    v13 = sub_2CAF220(a3, (__int64 *)&v54, a4);
    v7 = a2;
    v50 = v13;
    if ( v54 != v56 )
    {
      _libc_free((unsigned __int64)v54);
      v7 = a2;
    }
  }
  v54 = v56;
  v40 = v7;
  v55 = 0x800000000LL;
  v14 = sub_2CB1590((_QWORD *)*v9, v7, (__int64 *)&v50, a6 + 8, (int *)a6, 8u);
  v16 = (unsigned int)v55;
  v17 = v40;
  v18 = (unsigned int)v55 + 1LL;
  if ( v18 > HIDWORD(v55) )
  {
    v38 = v40;
    v42 = v14;
    sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 8u, v18, v15);
    v16 = (unsigned int)v55;
    v17 = v38;
    v14 = v42;
  }
  v54[v16] = v14;
  LODWORD(v55) = v55 + 1;
  v19 = sub_2CB1590((_QWORD *)*v9, v17, (__int64 *)&v50, a6 + 56, (int *)(a6 + 4), 8u);
  v21 = (unsigned int)v55;
  v22 = (unsigned int)v55 + 1LL;
  if ( v22 > HIDWORD(v55) )
  {
    v43 = v19;
    sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 8u, v22, v20);
    v21 = (unsigned int)v55;
    v19 = v43;
  }
  v54[v21] = v19;
  LODWORD(v55) = v55 + 1;
  v25 = sub_2CAF420(v44, a6, a7, (__int64 *)&v50);
  v26 = (unsigned int)v55;
  v27 = (unsigned int)v55 + 1LL;
  if ( v27 > HIDWORD(v55) )
  {
    sub_C8D5F0((__int64)&v54, v56, v27, 8u, v23, v24);
    v26 = (unsigned int)v55;
  }
  v54[v26] = v25;
  LODWORD(v55) = v55 + 1;
  if ( *(_DWORD *)a6 == 2 )
    v28 = (*(_DWORD *)(a6 + 4) == 2) + 8880;
  else
    v28 = (*(_DWORD *)(a6 + 4) == 2) + 8878;
  v29 = 0;
  v30 = sub_B6E160(v9, v28, 0, 0);
  v53 = 1;
  v52 = 3;
  v51 = "idp4a";
  v45 = v54;
  v41 = (unsigned int)v55;
  if ( v30 )
    v29 = *(_QWORD *)(v30 + 24);
  v37 = v30;
  v39 = v55 + 1;
  v31 = sub_BD2C40(88, (int)v55 + 1);
  v32 = (__int64)v31;
  if ( v31 )
  {
    sub_B44260((__int64)v31, **(_QWORD **)(v29 + 16), 56, v39 & 0x7FFFFFF, 0, 0);
    *(_QWORD *)(v32 + 72) = 0;
    sub_B4A290(v32, v29, v37, v45, v41, (__int64)&v51, 0, 0);
  }
  sub_B43DD0(v32, (__int64)v50);
  v33 = *(_DWORD *)(a6 + 112);
  *(_QWORD *)a1 = v32;
  v34 = *(_QWORD *)a6 == 0x200000002LL;
  *(_QWORD *)(a1 + 16) = v32;
  v50 = (_BYTE *)v32;
  *(_DWORD *)(a1 + 12) = v33;
  *(_DWORD *)(a1 + 8) = v34 + 1;
  if ( v33 != *(_DWORD *)(a7 + 12) )
  {
    sub_2CB0BD0((__int64)&v48, a3, a4, a1, a7);
    v35 = _mm_loadu_si128(&v48);
    *(_QWORD *)(a1 + 16) = v49;
    *(__m128i *)a1 = v35;
  }
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  return a1;
}
