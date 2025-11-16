// Function: sub_1D5EAC0
// Address: 0x1d5eac0
//
__int64 __fastcall sub_1D5EAC0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  _QWORD *v4; // r12
  __int64 v5; // r13
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // r15d
  char v10; // di
  __int64 v11; // r15
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // [rsp-A8h] [rbp-A8h]
  int v25; // [rsp-A0h] [rbp-A0h]
  unsigned int v26; // [rsp-98h] [rbp-98h]
  int v27; // [rsp-98h] [rbp-98h]
  __int64 v28; // [rsp-90h] [rbp-90h]
  char v29; // [rsp-7Dh] [rbp-7Dh] BYREF
  _BYTE v30[4]; // [rsp-7Ch] [rbp-7Ch] BYREF
  __int64 v31; // [rsp-78h] [rbp-78h] BYREF
  __int64 v32; // [rsp-70h] [rbp-70h]
  const void *v33; // [rsp-68h] [rbp-68h] BYREF
  __int64 v34; // [rsp-60h] [rbp-60h]
  __int64 v35; // [rsp-58h] [rbp-58h] BYREF
  __int64 v36; // [rsp-50h] [rbp-50h]
  __int64 v37; // [rsp-48h] [rbp-48h]

  if ( !*(_QWORD *)(a1 + 176) )
    return 0;
  if ( !*(_QWORD *)(a1 + 904) )
    return 0;
  v3 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(__int64 **)(a2 - 8)
     : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v4 = (_QWORD *)*v3;
  v5 = *(_QWORD *)*v3;
  v6 = (__int64 *)sub_16498A0(*v3);
  v28 = *(_QWORD *)(a1 + 176);
  v7 = sub_1D5D7E0(*(_QWORD *)(a1 + 904), (__int64 *)v5, 0);
  v9 = v8;
  v31 = v7;
  v32 = v8;
  if ( (_BYTE)v7 )
  {
    v10 = *(_BYTE *)(v28 + (unsigned __int8)v7 + 1155);
  }
  else if ( (unsigned __int8)sub_1F58D20(&v31) )
  {
    LOBYTE(v35) = 0;
    v36 = 0;
    v30[0] = 0;
    sub_1F426C0(v28, (_DWORD)v6, v31, v9, (unsigned int)&v35, (unsigned int)&v33, (__int64)v30);
    v10 = v30[0];
  }
  else
  {
    sub_1F40D10(&v35, v28, v6, v31, v32);
    LOBYTE(v33) = v36;
    v34 = v37;
    if ( (_BYTE)v36 )
    {
      v10 = *(_BYTE *)(v28 + (unsigned __int8)v36 + 1155);
    }
    else
    {
      v27 = v37;
      if ( (unsigned __int8)sub_1F58D20(&v33) )
      {
        LOBYTE(v35) = 0;
        v36 = 0;
        v29 = 0;
        sub_1F426C0(v28, (_DWORD)v6, (_DWORD)v33, v27, (unsigned int)&v35, (unsigned int)v30, (__int64)&v29);
        v10 = v29;
      }
      else
      {
        sub_1F40D10(&v35, v28, v6, v33, v34);
        v10 = sub_1D5E9F0(v28, (__int64)v6, (unsigned __int8)v36, v37);
      }
    }
  }
  v26 = sub_1D5A920(v10);
  if ( v26 <= *(_DWORD *)(v5 + 8) >> 8 )
    return 0;
  v11 = sub_1644C60(v6, v26);
  if ( *((_BYTE *)v4 + 16) == 17 && (unsigned __int8)sub_15E0520((__int64)v4) )
    v25 = 38;
  else
    v25 = 37;
  LOWORD(v37) = 257;
  v13 = sub_15FDBD0(v25, (__int64)v4, v11, (__int64)&v35, 0);
  sub_15F2120(v13, a2);
  v14 = (_QWORD *)sub_13CF970(a2);
  sub_1593B40(v14, v13);
  v24 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 != 1 )
  {
    v15 = 0;
    do
    {
      ++v15;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v16 = *(_QWORD *)(a2 - 8);
      else
        v16 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v17 = *(_QWORD *)(v16 + 24LL * (unsigned int)(2 * v15));
      LODWORD(v34) = *(_DWORD *)(v17 + 32);
      if ( (unsigned int)v34 > 0x40 )
        sub_16A4FD0((__int64)&v33, (const void **)(v17 + 24));
      else
        v33 = *(const void **)(v17 + 24);
      if ( v25 == 37 )
        sub_16A5C50((__int64)&v35, &v33, v26);
      else
        sub_16A5B10((__int64)&v35, &v33, v26);
      v18 = sub_159C0E0(v6, (__int64)&v35);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v19 = *(_QWORD *)(a2 - 8);
      else
        v19 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v20 = (__int64 *)(24LL * (unsigned int)(2 * v15) + v19);
      if ( *v20 )
      {
        v21 = v20[1];
        v22 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v22 = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
      }
      *v20 = v18;
      if ( v18 )
      {
        v23 = *(_QWORD *)(v18 + 8);
        v20[1] = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v23 + 16) & 3LL;
        v20[2] = (v18 + 8) | v20[2] & 3;
        *(_QWORD *)(v18 + 8) = v20;
      }
      if ( (unsigned int)v36 > 0x40 && v35 )
        j_j___libc_free_0_0(v35);
      if ( (unsigned int)v34 > 0x40 )
      {
        if ( v33 )
          j_j___libc_free_0_0(v33);
      }
    }
    while ( v24 != v15 );
  }
  return 1;
}
