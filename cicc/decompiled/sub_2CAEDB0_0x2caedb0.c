// Function: sub_2CAEDB0
// Address: 0x2caedb0
//
_QWORD *__fastcall sub_2CAEDB0(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rsi
  bool v11; // bl
  _QWORD *v12; // rsi
  _QWORD *v13; // rdx
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-1B0h]
  __int64 v17; // [rsp+8h] [rbp-1A8h]
  _QWORD v18[9]; // [rsp+10h] [rbp-1A0h] BYREF
  _QWORD v19[8]; // [rsp+58h] [rbp-158h] BYREF
  int v20; // [rsp+98h] [rbp-118h]
  __int64 v21; // [rsp+A0h] [rbp-110h]
  __int64 v22; // [rsp+A8h] [rbp-108h]
  __int64 v23; // [rsp+B0h] [rbp-100h]
  int v24; // [rsp+B8h] [rbp-F8h]
  __int64 v25; // [rsp+C0h] [rbp-F0h]
  __int64 v26; // [rsp+C8h] [rbp-E8h]
  __int64 v27; // [rsp+D0h] [rbp-E0h]
  __int64 v28; // [rsp+D8h] [rbp-D8h]
  __int64 v29; // [rsp+E0h] [rbp-D0h]
  __int64 v30; // [rsp+E8h] [rbp-C8h]
  __int64 v31; // [rsp+F0h] [rbp-C0h]
  __int64 v32; // [rsp+F8h] [rbp-B8h]
  __int64 v33; // [rsp+100h] [rbp-B0h]
  __int64 v34; // [rsp+108h] [rbp-A8h]
  __int64 v35; // [rsp+110h] [rbp-A0h]
  _QWORD v36[6]; // [rsp+120h] [rbp-90h] BYREF
  int v37; // [rsp+150h] [rbp-60h] BYREF
  __int64 v38; // [rsp+158h] [rbp-58h]
  int *v39; // [rsp+160h] [rbp-50h]
  int *v40; // [rsp+168h] [rbp-48h]
  __int64 v41; // [rsp+170h] [rbp-40h]

  v17 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v6 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v7 = sub_BC1CD0(a4, &unk_5035D48, a3);
  v8 = v7 + 8;
  if ( (unsigned int)qword_5012E48 | (unsigned int)qword_5012D68 )
  {
    v16 = v7 + 8;
    v15 = sub_BC1CD0(a4, &unk_4F881D0, a3);
    v8 = v16;
    v9 = v15 + 8;
  }
  else
  {
    v9 = 0;
  }
  v26 = v9;
  v10 = *a2;
  v19[2] = v19;
  v27 = v17;
  v19[3] = v19;
  v29 = v8;
  v25 = v10;
  v28 = v6;
  memset(v18, 0, 64);
  v19[0] = 0;
  v19[1] = 0;
  memset(&v19[4], 0, 32);
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v36[2] = v36;
  v36[3] = v36;
  v35 = 0;
  v36[0] = 0;
  v36[1] = 0;
  v36[4] = 0;
  v37 = 0;
  v38 = 0;
  v39 = &v37;
  v40 = &v37;
  v41 = 0;
  v11 = sub_2CABB50((__int64)v18, a3);
  sub_2C91DD0((__int64)v18);
  v12 = a1 + 4;
  v13 = a1 + 10;
  if ( v11 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v12;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v13;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v12;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v13;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}
