// Function: sub_19C0AD0
// Address: 0x19c0ad0
//
__int64 __fastcall sub_19C0AD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6, __int64 a7)
{
  _QWORD *v10; // r14
  _QWORD **v11; // rax
  _QWORD *v12; // r15
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // r12
  unsigned __int64 *v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 **v26; // rax
  __int64 *v27; // rsi
  __int64 v28; // rdx
  __int64 v30; // r14
  _QWORD *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rax
  unsigned __int64 *v34; // [rsp+0h] [rbp-E0h]
  char v35; // [rsp+Fh] [rbp-D1h]
  __int64 v36; // [rsp+10h] [rbp-D0h]
  __int64 v40; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int8 *v41[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v42; // [rsp+50h] [rbp-90h]
  unsigned __int8 *v43; // [rsp+60h] [rbp-80h] BYREF
  __int64 v44; // [rsp+68h] [rbp-78h]
  unsigned __int64 *v45; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v46; // [rsp+78h] [rbp-68h]
  unsigned __int64 *v47; // [rsp+80h] [rbp-60h]
  unsigned __int64 v48; // [rsp+88h] [rbp-58h]
  __int64 v49; // [rsp+90h] [rbp-50h]
  __int64 v50; // [rsp+98h] [rbp-48h]

  if ( *(_BYTE *)(a3 + 16) == 13 && (v30 = *(_QWORD *)a3, v31 = (_QWORD *)sub_16498A0(a2), v30 == sub_1643320(v31)) )
  {
    v32 = (__int64 *)sub_16498A0(a3);
    if ( a3 == sub_159C4F0(v32) )
    {
      v35 = 0;
      v10 = (_QWORD *)a2;
    }
    else
    {
      v33 = a4;
      v10 = (_QWORD *)a2;
      v35 = 1;
      a4 = a5;
      a5 = v33;
    }
  }
  else
  {
    LOWORD(v45) = 257;
    v10 = sub_1648A60(56, 2u);
    if ( v10 )
    {
      v11 = *(_QWORD ***)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      {
        v12 = v11[4];
        v13 = (__int64 *)sub_1643320(*v11);
        v14 = (__int64)sub_16463B0(v13, (unsigned int)v12);
      }
      else
      {
        v14 = sub_1643320(*v11);
      }
      sub_15FEC10((__int64)v10, v14, 51, 32, a2, a3, (__int64)&v43, (__int64)a6);
    }
    v35 = 0;
  }
  v36 = *(a6 - 3);
  v34 = (unsigned __int64 *)a6[5];
  v15 = sub_16498A0((__int64)a6);
  v16 = (unsigned __int8 *)a6[6];
  v43 = 0;
  v46 = v15;
  v17 = a6[5];
  v47 = 0;
  v44 = v17;
  LODWORD(v48) = 0;
  v49 = 0;
  v50 = 0;
  v45 = a6 + 3;
  v41[0] = v16;
  if ( v16 )
  {
    sub_1623A60((__int64)v41, (__int64)v16, 2);
    if ( v43 )
      sub_161E7C0((__int64)&v43, (__int64)v43);
    v43 = v41[0];
    if ( v41[0] )
      sub_1623210((__int64)v41, v41[0], (__int64)&v43);
  }
  v18 = sub_1648A60(56, 3u);
  v19 = v18;
  if ( v18 )
    sub_15F83E0((__int64)v18, a4, a5, (__int64)v10, 0);
  if ( a7 )
  {
    v41[1] = (unsigned __int8 *)14;
    v41[0] = (unsigned __int8 *)0xF00000002LL;
    sub_15F4370((__int64)v19, a7, (int *)v41, 4);
  }
  v42 = 257;
  if ( v44 )
  {
    v20 = v45;
    sub_157E9D0(v44 + 40, (__int64)v19);
    v21 = v19[3];
    v22 = *v20;
    v19[4] = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    v19[3] = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v19 + 3;
    *v20 = *v20 & 7 | (unsigned __int64)(v19 + 3);
  }
  sub_164B780((__int64)v19, (__int64 *)v41);
  if ( v43 )
  {
    v40 = (__int64)v43;
    sub_1623A60((__int64)&v40, (__int64)v43, 2);
    v23 = v19[6];
    if ( v23 )
      sub_161E7C0((__int64)(v19 + 6), v23);
    v24 = (unsigned __int8 *)v40;
    v19[6] = v40;
    if ( v24 )
      sub_1623210((__int64)&v40, v24, (__int64)(v19 + 6));
    if ( v43 )
      sub_161E7C0((__int64)&v43, (__int64)v43);
  }
  if ( v35 )
    sub_15F34F0((__int64)v19);
  sub_15F2070(a6);
  v25 = *(_QWORD *)(a1 + 304);
  if ( v25 )
  {
    v44 = 0x300000000LL;
    v43 = (unsigned __int8 *)&v45;
    if ( v36 == a4 )
    {
      if ( v36 == a5 )
      {
        v28 = 0;
        v27 = (__int64 *)&v45;
      }
      else
      {
        v27 = (__int64 *)&v45;
        v46 = a5 & 0xFFFFFFFFFFFFFFFBLL;
        v45 = v34;
        v28 = (unsigned int)(v44 + 1);
        LODWORD(v44) = v44 + 1;
      }
    }
    else
    {
      LODWORD(v44) = 1;
      v46 = a4 & 0xFFFFFFFFFFFFFFFBLL;
      v45 = v34;
      if ( v36 == a5 )
      {
        v28 = 1;
        v27 = (__int64 *)&v45;
      }
      else
      {
        v47 = v34;
        LODWORD(v44) = 2;
        v48 = a5 & 0xFFFFFFFFFFFFFFFBLL;
        v26 = (unsigned __int64 **)(v43 + 32);
        *((_QWORD *)v43 + 5) = v36 | 4;
        *v26 = v34;
        v27 = (__int64 *)v43;
        v28 = (unsigned int)(v44 + 1);
        LODWORD(v44) = v44 + 1;
        v25 = *(_QWORD *)(a1 + 304);
      }
    }
    sub_15DC140(v25, v27, v28);
    if ( v43 != (unsigned __int8 *)&v45 )
      _libc_free((unsigned __int64)v43);
    v25 = *(_QWORD *)(a1 + 304);
  }
  v43 = (unsigned __int8 *)v25;
  LODWORD(v45) = (_DWORD)&loc_1010000;
  v44 = *(_QWORD *)(a1 + 160);
  sub_1AAC5F0(v19, 0, &v43);
  return sub_1AAC5F0(v19, 1, &v43);
}
