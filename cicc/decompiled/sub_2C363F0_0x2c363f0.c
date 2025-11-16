// Function: sub_2C363F0
// Address: 0x2c363f0
//
void __fastcall sub_2C363F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  int v4; // eax
  __int64 v5; // rcx
  __int64 *v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 *v21; // rdx
  int v22; // r15d
  unsigned __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-A68h]
  _QWORD *v25; // [rsp+10h] [rbp-A60h]
  _QWORD v26[3]; // [rsp+20h] [rbp-A50h] BYREF
  int v27; // [rsp+38h] [rbp-A38h]
  char v28; // [rsp+3Ch] [rbp-A34h]
  char v29; // [rsp+40h] [rbp-A30h] BYREF
  _BYTE *v30; // [rsp+80h] [rbp-9F0h] BYREF
  __int64 v31; // [rsp+88h] [rbp-9E8h]
  _BYTE v32[320]; // [rsp+90h] [rbp-9E0h] BYREF
  _QWORD v33[54]; // [rsp+1D0h] [rbp-8A0h] BYREF
  _BYTE v34[32]; // [rsp+380h] [rbp-6F0h] BYREF
  _BYTE v35[64]; // [rsp+3A0h] [rbp-6D0h] BYREF
  _BYTE v36[8]; // [rsp+3E0h] [rbp-690h] BYREF
  int v37; // [rsp+3E8h] [rbp-688h]
  _BYTE v38[32]; // [rsp+530h] [rbp-540h] BYREF
  _BYTE v39[64]; // [rsp+550h] [rbp-520h] BYREF
  _QWORD v40[2]; // [rsp+590h] [rbp-4E0h] BYREF
  char v41; // [rsp+5A0h] [rbp-4D0h] BYREF
  _BYTE v42[32]; // [rsp+6E0h] [rbp-390h] BYREF
  _BYTE v43[64]; // [rsp+700h] [rbp-370h] BYREF
  _BYTE v44[8]; // [rsp+740h] [rbp-330h] BYREF
  unsigned int v45; // [rsp+748h] [rbp-328h]
  unsigned __int64 v46[4]; // [rsp+890h] [rbp-1E0h] BYREF
  _BYTE v47[64]; // [rsp+8B0h] [rbp-1C0h] BYREF
  _QWORD v48[2]; // [rsp+8F0h] [rbp-180h] BYREF
  char v49; // [rsp+900h] [rbp-170h] BYREF

  memset(v33, 0, sizeof(v33));
  LODWORD(v33[2]) = 8;
  v33[1] = &v33[4];
  v33[12] = &v33[14];
  v26[1] = &v29;
  BYTE4(v33[3]) = 1;
  HIDWORD(v33[13]) = 8;
  v26[0] = 0;
  v26[2] = 8;
  v27 = 0;
  v28 = 1;
  v30 = v32;
  v31 = 0x800000000LL;
  sub_AE6EC0((__int64)v26, a2);
  v3 = 1;
  if ( *(_BYTE *)(a2 + 8) )
  {
    v18 = a2;
    while ( 1 )
    {
      v19 = *(_DWORD *)(v18 + 88);
      if ( v19 )
        break;
      v18 = *(_QWORD *)(v18 + 48);
      if ( !v18 )
      {
        v3 = 0;
        goto LABEL_2;
      }
    }
    v3 = v19;
  }
LABEL_2:
  v4 = v31;
  if ( HIDWORD(v31) <= (unsigned int)v31 )
  {
    v24 = v3;
    v20 = (_QWORD *)sub_C8D7D0((__int64)&v30, (__int64)v32, 0, 0x28u, v46, v3);
    v21 = &v20[5 * (unsigned int)v31];
    if ( v21 )
    {
      *v21 = a2;
      v21[2] = a2;
      v21[1] = v24;
      v21[3] = 0;
      v21[4] = a2;
    }
    a2 = (__int64)v20;
    v25 = v20;
    sub_2BF6E30((__int64)&v30, v20);
    v22 = v46[0];
    v23 = (unsigned __int64)v25;
    if ( v30 != v32 )
    {
      _libc_free((unsigned __int64)v30);
      v23 = (unsigned __int64)v25;
    }
    LODWORD(v31) = v31 + 1;
    v30 = (_BYTE *)v23;
    HIDWORD(v31) = v22;
  }
  else
  {
    v5 = 5LL * (unsigned int)v31;
    v6 = (__int64 *)&v30[40 * (unsigned int)v31];
    if ( v6 )
    {
      *v6 = a2;
      v6[1] = v3;
      v6[2] = a2;
      v6[3] = 0;
      v6[4] = a2;
      v4 = v31;
    }
    LODWORD(v31) = v4 + 1;
  }
  sub_2BF6FC0((__int64)v26, a2, (__int64)v6, v5, v2, v3);
  sub_2BF6E90((__int64)v42, (__int64)v33, v7, v8, v9, v10);
  sub_C8CF70((__int64)v46, v47, 8, (__int64)v43, (__int64)v42);
  v14 = v45;
  v48[0] = &v49;
  v48[1] = 0x800000000LL;
  if ( v45 )
    sub_2C36200((__int64)v48, (__int64)v44, v45, v11, v12, v13);
  sub_2BF6E90((__int64)v34, (__int64)v26, v14, v11, v12, v13);
  sub_C8CF70((__int64)v38, v39, 8, (__int64)v35, (__int64)v34);
  v40[0] = &v41;
  v40[1] = 0x800000000LL;
  if ( v37 )
    sub_2C36200((__int64)v40, (__int64)v36, v15, v16, v17, (__int64)v38);
  sub_2BF7440((__int64)v38, (__int64)v46, a1, v16, v17, (__int64)v38);
  sub_2C2BCB0((__int64)v38);
  sub_2C2BCB0((__int64)v34);
  sub_2C2BCB0((__int64)v46);
  sub_2C2BCB0((__int64)v42);
  sub_2C2BCB0((__int64)v26);
  sub_2C2BCB0((__int64)v33);
}
