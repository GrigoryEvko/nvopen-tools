// Function: sub_2713AA0
// Address: 0x2713aa0
//
__int64 __fastcall sub_2713AA0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _BYTE v10[16]; // [rsp+0h] [rbp-170h] BYREF
  __int64 v11; // [rsp+10h] [rbp-160h]
  char v12; // [rsp+20h] [rbp-150h]
  __int64 v13; // [rsp+30h] [rbp-140h]
  __int64 v14; // [rsp+38h] [rbp-138h]
  __int64 v15; // [rsp+40h] [rbp-130h]
  __int64 v16; // [rsp+48h] [rbp-128h] BYREF
  _BYTE *v17; // [rsp+50h] [rbp-120h]
  __int64 v18; // [rsp+58h] [rbp-118h]
  int v19; // [rsp+60h] [rbp-110h]
  char v20; // [rsp+64h] [rbp-10Ch]
  _BYTE v21[16]; // [rsp+68h] [rbp-108h] BYREF
  __int64 v22; // [rsp+78h] [rbp-F8h] BYREF
  _BYTE *v23; // [rsp+80h] [rbp-F0h]
  __int64 v24; // [rsp+88h] [rbp-E8h]
  int v25; // [rsp+90h] [rbp-E0h]
  char v26; // [rsp+94h] [rbp-DCh]
  _BYTE v27[16]; // [rsp+98h] [rbp-D8h] BYREF
  char v28; // [rsp+A8h] [rbp-C8h]
  __int64 v29; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v30[3]; // [rsp+B8h] [rbp-B8h] BYREF
  _BYTE v31[8]; // [rsp+D0h] [rbp-A0h] BYREF
  unsigned __int64 v32; // [rsp+D8h] [rbp-98h]
  char v33; // [rsp+ECh] [rbp-84h]
  _BYTE v34[16]; // [rsp+F0h] [rbp-80h] BYREF
  _BYTE v35[8]; // [rsp+100h] [rbp-70h] BYREF
  unsigned __int64 v36; // [rsp+108h] [rbp-68h]
  char v37; // [rsp+11Ch] [rbp-54h]
  _BYTE v38[80]; // [rsp+120h] [rbp-50h] BYREF

  v2 = *a2;
  v30[0] = 0;
  v29 = v2;
  sub_2712ED0((__int64)v10, a1, &v29, v30);
  if ( !v12 )
    return *(_QWORD *)(a1 + 32) + 136LL * *(_QWORD *)(v11 + 8) + 8;
  v4 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
  v13 = 0;
  *(_QWORD *)(v11 + 8) = 0xF0F0F0F0F0F0F0F1LL * (v4 >> 3);
  v5 = *a2;
  v17 = v21;
  v29 = v5;
  LOWORD(v30[0]) = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v18 = 2;
  v19 = 0;
  v20 = 1;
  v22 = 0;
  v23 = v27;
  v24 = 2;
  v25 = 0;
  v26 = 1;
  v28 = 0;
  BYTE2(v30[0]) = 0;
  v30[1] = 0;
  v30[2] = 0;
  sub_C8CF70((__int64)v31, v34, 2, (__int64)v21, (__int64)&v16);
  sub_C8CF70((__int64)v35, v38, 2, (__int64)v27, (__int64)&v22);
  v38[16] = v28;
  sub_2712480((unsigned __int64 *)(a1 + 32), (__int64)&v29, v6, v7, v8, v9);
  if ( v37 )
  {
    if ( v33 )
      goto LABEL_5;
  }
  else
  {
    _libc_free(v36);
    if ( v33 )
    {
LABEL_5:
      if ( v26 )
        goto LABEL_6;
LABEL_10:
      _libc_free((unsigned __int64)v23);
      if ( v20 )
        return v4 + *(_QWORD *)(a1 + 32) + 8;
LABEL_11:
      _libc_free((unsigned __int64)v17);
      return v4 + *(_QWORD *)(a1 + 32) + 8;
    }
  }
  _libc_free(v32);
  if ( !v26 )
    goto LABEL_10;
LABEL_6:
  if ( !v20 )
    goto LABEL_11;
  return v4 + *(_QWORD *)(a1 + 32) + 8;
}
