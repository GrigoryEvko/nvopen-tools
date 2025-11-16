// Function: sub_18D2800
// Address: 0x18d2800
//
__int64 __fastcall sub_18D2800(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // [rsp+8h] [rbp-198h]
  _BYTE v7[16]; // [rsp+10h] [rbp-190h] BYREF
  __int64 v8; // [rsp+20h] [rbp-180h]
  char v9; // [rsp+30h] [rbp-170h]
  __int64 v10; // [rsp+40h] [rbp-160h]
  __int64 v11; // [rsp+48h] [rbp-158h]
  __int64 v12; // [rsp+50h] [rbp-150h]
  __int64 v13; // [rsp+58h] [rbp-148h] BYREF
  _BYTE *v14; // [rsp+60h] [rbp-140h]
  _BYTE *v15; // [rsp+68h] [rbp-138h]
  __int64 v16; // [rsp+70h] [rbp-130h]
  int v17; // [rsp+78h] [rbp-128h]
  _BYTE v18[16]; // [rsp+80h] [rbp-120h] BYREF
  __int64 v19; // [rsp+90h] [rbp-110h] BYREF
  _BYTE *v20; // [rsp+98h] [rbp-108h]
  _BYTE *v21; // [rsp+A0h] [rbp-100h]
  __int64 v22; // [rsp+A8h] [rbp-F8h]
  int v23; // [rsp+B0h] [rbp-F0h]
  _BYTE v24[16]; // [rsp+B8h] [rbp-E8h] BYREF
  char v25; // [rsp+C8h] [rbp-D8h]
  __int64 v26; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+D8h] [rbp-C8h] BYREF
  __int64 v28; // [rsp+E0h] [rbp-C0h]
  __int64 v29; // [rsp+E8h] [rbp-B8h]
  _QWORD v30[2]; // [rsp+F0h] [rbp-B0h] BYREF
  unsigned __int64 v31; // [rsp+100h] [rbp-A0h]
  _BYTE v32[16]; // [rsp+118h] [rbp-88h] BYREF
  _QWORD v33[2]; // [rsp+128h] [rbp-78h] BYREF
  unsigned __int64 v34; // [rsp+138h] [rbp-68h]
  _BYTE v35[16]; // [rsp+150h] [rbp-50h] BYREF
  char v36; // [rsp+160h] [rbp-40h]

  v2 = *a2;
  v27 = 0;
  v26 = v2;
  sub_18D2550((__int64)v7, (__int64)a1, &v26, &v27);
  if ( !v9 )
    return a1[4] + 152LL * *(_QWORD *)(v8 + 8) + 8;
  v6 = a1[5] - a1[4];
  v10 = 0;
  *(_QWORD *)(v8 + 8) = 0x86BCA1AF286BCA1BLL * (v6 >> 3);
  v14 = v18;
  v15 = v18;
  v20 = v24;
  v21 = v24;
  v4 = *a2;
  v11 = 0;
  v26 = v4;
  LOWORD(v27) = 0;
  v12 = 0;
  v13 = 0;
  v16 = 2;
  v17 = 0;
  v19 = 0;
  v22 = 2;
  v23 = 0;
  v25 = 0;
  BYTE2(v27) = 0;
  v28 = 0;
  v29 = 0;
  sub_16CCEE0(v30, (__int64)v32, 2, (__int64)&v13);
  sub_16CCEE0(v33, (__int64)v35, 2, (__int64)&v19);
  v5 = a1[5];
  v36 = v25;
  if ( v5 == a1[6] )
  {
    sub_18D16D0(a1 + 4, (char *)v5, &v26);
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = v26;
      *(_WORD *)(v5 + 8) = v27;
      *(_BYTE *)(v5 + 10) = BYTE2(v27);
      *(_WORD *)(v5 + 16) = v28;
      *(_QWORD *)(v5 + 24) = v29;
      sub_16CCEE0((_QWORD *)(v5 + 32), v5 + 72, 2, (__int64)v30);
      sub_16CCEE0((_QWORD *)(v5 + 88), v5 + 128, 2, (__int64)v33);
      *(_BYTE *)(v5 + 144) = v36;
      v5 = a1[5];
    }
    a1[5] = v5 + 152;
  }
  if ( v34 != v33[1] )
    _libc_free(v34);
  if ( v31 != v30[1] )
    _libc_free(v31);
  if ( v21 != v20 )
    _libc_free((unsigned __int64)v21);
  if ( v15 != v14 )
    _libc_free((unsigned __int64)v15);
  return a1[4] + v6 + 8;
}
