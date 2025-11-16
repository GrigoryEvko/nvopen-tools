// Function: sub_1D7D980
// Address: 0x1d7d980
//
__int64 __fastcall sub_1D7D980(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r15
  char v5; // al
  unsigned __int64 *v6; // r13
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  char v11; // al
  __int64 v12; // rcx
  char v13; // r9
  _QWORD *v14; // rax
  __int64 result; // rax
  unsigned int v16; // esi
  int v17; // edx
  int v18; // edx
  unsigned __int64 v19; // rdx
  bool v20; // zf
  __int64 v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-158h]
  __int64 v23; // [rsp+8h] [rbp-158h]
  __int64 v24; // [rsp+8h] [rbp-158h]
  __int64 v25; // [rsp+8h] [rbp-158h]
  _QWORD *v26; // [rsp+8h] [rbp-158h]
  _QWORD *v27; // [rsp+8h] [rbp-158h]
  __int64 v28; // [rsp+10h] [rbp-150h]
  unsigned __int64 v29[2]; // [rsp+20h] [rbp-140h] BYREF
  unsigned __int64 v30; // [rsp+30h] [rbp-130h]
  __int64 v31; // [rsp+40h] [rbp-120h]
  unsigned __int64 v32[2]; // [rsp+48h] [rbp-118h] BYREF
  unsigned __int64 v33; // [rsp+58h] [rbp-108h]
  void *v34; // [rsp+60h] [rbp-100h] BYREF
  unsigned __int64 v35[2]; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v36; // [rsp+78h] [rbp-E8h]
  __int64 v37; // [rsp+80h] [rbp-E0h]
  void *v38; // [rsp+90h] [rbp-D0h]
  _QWORD v39[2]; // [rsp+98h] [rbp-C8h] BYREF
  __int64 v40; // [rsp+A8h] [rbp-B8h]
  __int64 v41; // [rsp+B0h] [rbp-B0h]
  _QWORD *v42; // [rsp+C0h] [rbp-A0h] BYREF
  _QWORD v43[2]; // [rsp+C8h] [rbp-98h] BYREF
  __int64 v44; // [rsp+D8h] [rbp-88h]
  __int64 v45; // [rsp+E0h] [rbp-80h]
  unsigned __int64 *v46; // [rsp+F0h] [rbp-70h] BYREF
  unsigned __int64 v47; // [rsp+F8h] [rbp-68h] BYREF
  __int64 v48; // [rsp+100h] [rbp-60h]
  __int64 v49; // [rsp+108h] [rbp-58h]
  unsigned __int64 v50; // [rsp+110h] [rbp-50h]
  unsigned __int64 v51[2]; // [rsp+118h] [rbp-48h] BYREF
  unsigned __int64 v52; // [rsp+128h] [rbp-38h]

  v3 = a1[1];
  v35[1] = 0;
  v35[0] = v3 & 6;
  v36 = a1[3];
  if ( v36 != 0 && v36 != -8 && v36 != -16 )
    sub_1649AC0(v35, v3 & 0xFFFFFFFFFFFFFFF8LL);
  v4 = a1[4];
  v37 = v4;
  v34 = &unk_49F9E38;
  v5 = sub_1D682F0(v4, (__int64)&v34, &v46);
  v6 = v46;
  if ( !v5 )
    v6 = (unsigned __int64 *)(*(_QWORD *)(v4 + 8) + ((unsigned __int64)*(unsigned int *)(v4 + 24) << 6));
  v7 = v37;
  if ( v6 != (unsigned __int64 *)(*(_QWORD *)(v37 + 8) + ((unsigned __int64)*(unsigned int *)(v37 + 24) << 6)) )
  {
    v8 = v6[7];
    v29[0] = 6;
    v29[1] = 0;
    v30 = v8;
    if ( v8 != 0 && v8 != -8 && v8 != -16 )
    {
      sub_1649AC0(v29, v6[5] & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v37;
    }
    v28 = v7;
    sub_1455FA0((__int64)(v6 + 5));
    v46 = (unsigned __int64 *)&unk_49F9E38;
    v47 = 2;
    v48 = 0;
    v49 = -16;
    v50 = 0;
    sub_1D5A8A0(v6 + 1, &v47);
    v6[4] = v50;
    v46 = (unsigned __int64 *)&unk_49EE2B0;
    sub_1455FA0((__int64)&v47);
    v31 = a2;
    v32[0] = 6;
    --*(_DWORD *)(v28 + 16);
    ++*(_DWORD *)(v28 + 20);
    v9 = v37;
    v33 = v30;
    v32[1] = 0;
    if ( v30 != -8 && v30 != 0 && v30 != -16 )
    {
      v22 = v37;
      sub_1649AC0(v32, v29[0] & 0xFFFFFFFFFFFFFFF8LL);
      a2 = v31;
      v9 = v22;
    }
    v40 = a2;
    v39[0] = 2;
    v39[1] = 0;
    if ( a2 == -8 || a2 == 0 || a2 == -16 )
    {
      v41 = v9;
      v47 = 2;
      v48 = 0;
      v38 = &unk_49F9E38;
      v10 = v9;
      v49 = a2;
    }
    else
    {
      v23 = v9;
      sub_164C220((__int64)v39);
      v38 = &unk_49F9E38;
      v9 = v23;
      v48 = 0;
      v41 = v23;
      v47 = v39[0] & 6;
      v49 = v40;
      if ( v40 == -8 || v40 == 0 || v40 == -16 )
      {
        v10 = v23;
      }
      else
      {
        sub_1649AC0(&v47, v39[0] & 0xFFFFFFFFFFFFFFF8LL);
        v10 = v41;
        v9 = v23;
      }
    }
    v50 = v10;
    v46 = (unsigned __int64 *)&unk_49F9E38;
    v51[0] = 6;
    v51[1] = 0;
    v52 = v33;
    if ( v33 != 0 && v33 != -8 && v33 != -16 )
    {
      v24 = v9;
      sub_1649AC0(v51, v32[0] & 0xFFFFFFFFFFFFFFF8LL);
      v9 = v24;
    }
    v25 = v9;
    v11 = sub_1D682F0(v9, (__int64)&v46, &v42);
    v12 = v25;
    v13 = v11;
    v14 = v42;
    if ( v13 )
      goto LABEL_23;
    v16 = *(_DWORD *)(v25 + 24);
    v17 = *(_DWORD *)(v25 + 16);
    ++*(_QWORD *)v25;
    v18 = v17 + 1;
    if ( 4 * v18 >= 3 * v16 )
    {
      v16 *= 2;
      v21 = v25;
    }
    else
    {
      if ( v16 - *(_DWORD *)(v25 + 20) - v18 > v16 >> 3 )
        goto LABEL_31;
      v21 = v25;
    }
    sub_1D73330(v21, v16);
    sub_1D682F0(v25, (__int64)&v46, &v42);
    v12 = v25;
    v14 = v42;
    v18 = *(_DWORD *)(v25 + 16) + 1;
LABEL_31:
    *(_DWORD *)(v12 + 16) = v18;
    v43[0] = 2;
    v43[1] = 0;
    v44 = -8;
    v45 = 0;
    if ( v14[3] != -8 )
    {
      --*(_DWORD *)(v12 + 20);
      v42 = &unk_49EE2B0;
      if ( v44 != 0 && v44 != -8 && v44 != -16 )
      {
        v26 = v14;
        sub_1649B30(v43);
        v14 = v26;
      }
    }
    v27 = v14;
    sub_1D5A8A0(v14 + 1, &v47);
    v27[4] = v50;
    v27[5] = 6;
    v27[6] = 0;
    v19 = v52;
    v20 = v52 == -8;
    v27[7] = v52;
    if ( v19 != 0 && !v20 && v19 != -16 )
      sub_1649AC0(v27 + 5, v51[0] & 0xFFFFFFFFFFFFFFF8LL);
LABEL_23:
    sub_1455FA0((__int64)v51);
    v46 = (unsigned __int64 *)&unk_49EE2B0;
    sub_1455FA0((__int64)&v47);
    v38 = &unk_49EE2B0;
    sub_1455FA0((__int64)v39);
    sub_1455FA0((__int64)v32);
    sub_1455FA0((__int64)v29);
  }
  v34 = &unk_49EE2B0;
  result = v36;
  if ( v36 != 0 && v36 != -8 && v36 != -16 )
    return sub_1649B30(v35);
  return result;
}
