// Function: sub_2956FA0
// Address: 0x2956fa0
//
__int64 __fastcall sub_2956FA0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v19; // [rsp+8h] [rbp-128h]
  __int64 v20; // [rsp+18h] [rbp-118h] BYREF
  __int64 v21; // [rsp+20h] [rbp-110h] BYREF
  _QWORD *v22; // [rsp+28h] [rbp-108h]
  int v23; // [rsp+30h] [rbp-100h]
  int v24; // [rsp+34h] [rbp-FCh]
  int v25; // [rsp+38h] [rbp-F8h]
  char v26; // [rsp+3Ch] [rbp-F4h]
  _QWORD v27[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE *v29; // [rsp+58h] [rbp-D8h]
  __int64 v30; // [rsp+60h] [rbp-D0h]
  int v31; // [rsp+68h] [rbp-C8h]
  char v32; // [rsp+6Ch] [rbp-C4h]
  _BYTE v33[16]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v34[6]; // [rsp+80h] [rbp-B0h] BYREF
  char v35; // [rsp+B0h] [rbp-80h]
  __int64 v36; // [rsp+B8h] [rbp-78h]
  _QWORD *v37; // [rsp+C0h] [rbp-70h]
  __int64 v38; // [rsp+C8h] [rbp-68h]
  unsigned int v39; // [rsp+D0h] [rbp-60h]
  __int64 v40; // [rsp+D8h] [rbp-58h]
  _QWORD *v41; // [rsp+E0h] [rbp-50h]
  __int64 v42; // [rsp+E8h] [rbp-48h]
  unsigned int v43; // [rsp+F0h] [rbp-40h]

  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v19 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v8 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v9 = *a2;
  v34[1] = v7 + 8;
  v20 = a4;
  v34[3] = v8 + 8;
  v34[4] = sub_2950610;
  v34[2] = v19 + 8;
  v34[0] = 0;
  v34[5] = &v20;
  v35 = v9;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  if ( (_BYTE)qword_5005988 || !(unsigned __int8)sub_2953BC0((__int64)v34, a3) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    v22 = v27;
    v27[0] = &unk_4F82408;
    v23 = 2;
    v25 = 0;
    v26 = 1;
    v28 = 0;
    v29 = v33;
    v30 = 2;
    v31 = 0;
    v32 = 1;
    v24 = 1;
    v21 = 1;
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v27, (__int64)&v21);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v33, (__int64)&v28);
    if ( !v32 )
      _libc_free((unsigned __int64)v29);
    if ( !v26 )
      _libc_free((unsigned __int64)v22);
  }
  v10 = v43;
  if ( v43 )
  {
    v11 = v41;
    v12 = &v41[6 * v43];
    while ( 1 )
    {
      while ( *v11 == -4096 )
      {
        if ( v11[1] != -4096 )
          goto LABEL_6;
        v11 += 6;
        if ( v12 == v11 )
        {
LABEL_12:
          v10 = v43;
          goto LABEL_13;
        }
      }
      if ( *v11 != -8192 || v11[1] != -8192 )
      {
LABEL_6:
        v13 = v11[2];
        if ( (_QWORD *)v13 != v11 + 4 )
          _libc_free(v13);
      }
      v11 += 6;
      if ( v12 == v11 )
        goto LABEL_12;
    }
  }
LABEL_13:
  sub_C7D6A0((__int64)v41, 48 * v10, 8);
  v14 = v39;
  if ( v39 )
  {
    v15 = v37;
    v16 = &v37[6 * v39];
    while ( 1 )
    {
      while ( *v15 == -4096 )
      {
        if ( v15[1] != -4096 )
          goto LABEL_16;
        v15 += 6;
        if ( v16 == v15 )
        {
LABEL_22:
          v14 = v39;
          goto LABEL_23;
        }
      }
      if ( *v15 != -8192 || v15[1] != -8192 )
      {
LABEL_16:
        v17 = v15[2];
        if ( (_QWORD *)v17 != v15 + 4 )
          _libc_free(v17);
      }
      v15 += 6;
      if ( v16 == v15 )
        goto LABEL_22;
    }
  }
LABEL_23:
  sub_C7D6A0((__int64)v37, 48 * v14, 8);
  return a1;
}
