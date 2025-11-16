// Function: sub_37074E0
// Address: 0x37074e0
//
__int64 *__fastcall sub_37074E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  _BYTE *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // r8d
  __int16 v10; // ax
  __int16 v11; // dx
  __int64 v12; // rax
  __int16 v14; // [rsp+4Ch] [rbp-174h] BYREF
  __int16 v15; // [rsp+4Eh] [rbp-172h]
  __int64 v16; // [rsp+50h] [rbp-170h] BYREF
  __int64 v17; // [rsp+58h] [rbp-168h] BYREF
  __int64 v18; // [rsp+60h] [rbp-160h] BYREF
  unsigned __int64 v19; // [rsp+68h] [rbp-158h] BYREF
  __int16 *v20; // [rsp+70h] [rbp-150h] BYREF
  __int64 v21; // [rsp+78h] [rbp-148h]
  __int64 v22; // [rsp+80h] [rbp-140h]
  void *v23; // [rsp+90h] [rbp-130h]
  int v24; // [rsp+98h] [rbp-128h]
  __int64 v25; // [rsp+A0h] [rbp-120h]
  __int64 v26; // [rsp+A8h] [rbp-118h]
  _QWORD v27[2]; // [rsp+B0h] [rbp-110h] BYREF
  volatile signed __int32 *v28; // [rsp+C0h] [rbp-100h]
  __int64 v29; // [rsp+C8h] [rbp-F8h]
  __int64 v30; // [rsp+D0h] [rbp-F0h]
  __int64 v31; // [rsp+D8h] [rbp-E8h]
  char v32; // [rsp+E0h] [rbp-E0h]
  __int64 v33; // [rsp+E8h] [rbp-D8h]
  _QWORD v34[2]; // [rsp+F0h] [rbp-D0h] BYREF
  void *v35; // [rsp+100h] [rbp-C0h] BYREF
  char v36; // [rsp+10Ah] [rbp-B6h]
  char v37; // [rsp+10Eh] [rbp-B2h]
  _BYTE *v38; // [rsp+110h] [rbp-B0h]
  __int64 v39; // [rsp+118h] [rbp-A8h]
  _BYTE v40[24]; // [rsp+120h] [rbp-A0h] BYREF
  _QWORD *v41; // [rsp+138h] [rbp-88h]
  __int64 v42; // [rsp+140h] [rbp-80h]
  __int64 v43; // [rsp+148h] [rbp-78h]
  __int64 v44; // [rsp+150h] [rbp-70h]
  int v45; // [rsp+158h] [rbp-68h]
  void *v46; // [rsp+160h] [rbp-60h] BYREF
  unsigned __int64 v47; // [rsp+168h] [rbp-58h] BYREF
  _BYTE *v48; // [rsp+170h] [rbp-50h]
  _BYTE *v49; // [rsp+178h] [rbp-48h]
  __int64 *v50; // [rsp+180h] [rbp-40h]

  v25 = a2;
  v26 = a3;
  v23 = &unk_49E6828;
  v24 = 1;
  sub_12548A0(v27);
  v15 = 4611;
  v34[0] = &unk_4A3C7C8;
  v34[1] = v27;
  v36 = 0;
  v35 = &unk_4A3C998;
  v38 = v40;
  v39 = 0x200000000LL;
  v20 = &v14;
  v37 = 0;
  v41 = v27;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v14 = 2;
  v21 = 4;
  sub_370EAB0(&v16, &v35, &v20);
  v5 = v16;
  v16 = 0;
  v17 = 0;
  v19 = v5 | 1;
  sub_3706DA0(&v18, (__int64 *)&v19);
  if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  v18 = 0;
  sub_9C66B0(&v18);
  sub_9C66B0((__int64 *)&v19);
  sub_9C66B0(&v17);
  sub_9C66B0(&v16);
  v47 = 0;
  v48 = 0;
  v46 = &unk_4A35300;
  v50 = (__int64 *)&v46;
  v49 = 0;
  v20 = (__int16 *)v34;
  sub_37071D0((__int64)&v47, 0, &v20);
  v6 = v48;
  v20 = (__int16 *)a4;
  if ( v49 == v48 )
  {
    sub_37071D0((__int64)&v47, v48, &v20);
  }
  else
  {
    if ( v48 )
    {
      *(_QWORD *)v48 = a4;
      v6 = v48;
    }
    v48 = v6 + 8;
  }
  while ( 1 )
  {
    if ( v32 )
    {
      v7 = v31;
    }
    else
    {
      v7 = 0;
      if ( v29 )
      {
        v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 40LL))(v29);
        v7 = v8 - v30;
      }
    }
    if ( v33 == v7 )
      break;
    v20 = 0;
    v21 = 0;
    sub_1254950(&v19, (__int64)v27, (__int64)&v20, 2u);
    if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v20 = (__int16 *)(v19 & 0xFFFFFFFFFFFFFFFELL | 1);
      *a1 = 0;
      sub_9C6670(a1, &v20);
      sub_9C66B0((__int64 *)&v20);
      goto LABEL_17;
    }
    v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 16LL))(v29);
    v10 = *v20;
    v21 = 0;
    v22 = 0;
    v11 = __ROL2__(v10, 8);
    if ( v9 != 1 )
      v10 = v11;
    LOWORD(v20) = v10;
    sub_3705770((__int64 *)&v19, (__int16 *)&v20, v50);
    if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v19 = v19 & 0xFFFFFFFFFFFFFFFELL | 1;
      *a1 = 0;
      sub_9C6670(a1, &v19);
      sub_9C66B0((__int64 *)&v19);
      goto LABEL_17;
    }
  }
  v20 = 0;
  *a1 = 1;
  sub_9C66B0((__int64 *)&v20);
LABEL_17:
  v46 = &unk_4A35300;
  if ( v47 )
    j_j___libc_free_0(v47);
  v21 = 4;
  v15 = 4611;
  v34[0] = &unk_4A3C7C8;
  v14 = 2;
  v20 = &v14;
  sub_370CE40(&v16, &v35, &v20);
  v12 = v16;
  v16 = 0;
  v17 = 0;
  v19 = v12 | 1;
  sub_3706DA0(&v18, (__int64 *)&v19);
  if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  v18 = 0;
  sub_9C66B0(&v18);
  sub_9C66B0((__int64 *)&v19);
  sub_9C66B0(&v17);
  sub_9C66B0(&v16);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  v27[0] = &unk_49E6870;
  if ( v28 )
    sub_A191D0(v28);
  return a1;
}
