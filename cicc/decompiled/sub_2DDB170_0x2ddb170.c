// Function: sub_2DDB170
// Address: 0x2ddb170
//
__int64 __fastcall sub_2DDB170(__int64 a1, __int64 *a2, __int64 a3)
{
  void *v4; // r15
  void *v5; // r13
  char v6; // al
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // edi
  __int16 v10; // cx
  __int64 v12; // [rsp+10h] [rbp-160h] BYREF
  _QWORD *v13; // [rsp+18h] [rbp-158h]
  int v14; // [rsp+20h] [rbp-150h]
  int v15; // [rsp+24h] [rbp-14Ch]
  int v16; // [rsp+28h] [rbp-148h]
  char v17; // [rsp+2Ch] [rbp-144h]
  _QWORD v18[2]; // [rsp+30h] [rbp-140h] BYREF
  __int64 v19; // [rsp+40h] [rbp-130h] BYREF
  _BYTE *v20; // [rsp+48h] [rbp-128h]
  __int64 v21; // [rsp+50h] [rbp-120h]
  int v22; // [rsp+58h] [rbp-118h]
  char v23; // [rsp+5Ch] [rbp-114h]
  _BYTE v24[16]; // [rsp+60h] [rbp-110h] BYREF
  _QWORD v25[2]; // [rsp+70h] [rbp-100h] BYREF
  int v26; // [rsp+80h] [rbp-F0h]
  __int16 v27; // [rsp+84h] [rbp-ECh]
  char v28; // [rsp+86h] [rbp-EAh]
  char v29; // [rsp+88h] [rbp-E8h]
  __int64 v30; // [rsp+90h] [rbp-E0h]
  __int64 v31; // [rsp+98h] [rbp-D8h]
  __int64 v32; // [rsp+A0h] [rbp-D0h]
  __int64 v33; // [rsp+A8h] [rbp-C8h]
  _BYTE *v34; // [rsp+B0h] [rbp-C0h]
  __int64 v35; // [rsp+B8h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v4 = (void *)(a1 + 32);
  v5 = (void *)(a1 + 80);
  v6 = *((_BYTE *)a2 + 22);
  v7 = *a2;
  v29 = 0;
  v8 = a2[1];
  v9 = *((_DWORD *)a2 + 4);
  v30 = 0;
  v10 = *((_WORD *)a2 + 10);
  v28 = v6;
  v25[0] = v7;
  v25[1] = v8;
  v26 = v9;
  v27 = v10;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = v36;
  v35 = 0x1000000000LL;
  if ( (_BYTE)qword_501E0E8 && (unsigned __int8)sub_2DDA340((__int64)v25, a3) )
  {
    v20 = v24;
    v13 = v18;
    v18[0] = &unk_4F82408;
    v14 = 2;
    v16 = 0;
    v17 = 1;
    v19 = 0;
    v21 = 2;
    v22 = 0;
    v23 = 1;
    v15 = 1;
    v12 = 1;
    sub_C8CF70(a1, v4, 2, (__int64)v18, (__int64)&v12);
    sub_C8CF70(a1 + 48, v5, 2, (__int64)v24, (__int64)&v19);
    if ( !v23 )
      _libc_free((unsigned __int64)v20);
    if ( !v17 )
      _libc_free((unsigned __int64)v13);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v5;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  sub_C7D6A0(v31, 8LL * (unsigned int)v33, 8);
  return a1;
}
