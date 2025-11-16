// Function: sub_D6E4B0
// Address: 0xd6e4b0
//
__int64 __fastcall sub_D6E4B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r13
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned __int8 *v10; // rdi
  int v11; // eax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  _QWORD *v18; // r13
  __int64 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // ecx
  _QWORD *v23; // rdx
  _QWORD *v24; // rcx
  int v25; // r12d
  __int64 v26; // rcx
  __int64 v27; // r15
  _BYTE *v28; // r12
  _QWORD *v29; // rbx
  unsigned int v30; // eax
  int v31; // eax
  unsigned __int64 v32; // r12
  unsigned int v33; // eax
  _QWORD *v34; // [rsp+0h] [rbp-240h]
  _QWORD *v35; // [rsp+0h] [rbp-240h]
  unsigned __int64 v36[2]; // [rsp+10h] [rbp-230h] BYREF
  __int64 v37; // [rsp+20h] [rbp-220h]
  __int64 v38; // [rsp+30h] [rbp-210h] BYREF
  _QWORD *v39; // [rsp+38h] [rbp-208h]
  __int64 v40; // [rsp+40h] [rbp-200h]
  __int64 v41; // [rsp+48h] [rbp-1F8h]
  _BYTE *v42; // [rsp+50h] [rbp-1F0h]
  __int64 v43; // [rsp+58h] [rbp-1E8h]
  _BYTE v44[32]; // [rsp+60h] [rbp-1E0h] BYREF
  _BYTE *v45; // [rsp+80h] [rbp-1C0h] BYREF
  __int64 v46; // [rsp+88h] [rbp-1B8h]
  _BYTE v47[432]; // [rsp+90h] [rbp-1B0h] BYREF

  v6 = a3;
  if ( *(_BYTE *)a2 == 28 )
  {
    v8 = sub_D67DF0((__int64 *)a2);
  }
  else
  {
    if ( *(_BYTE *)a2 == 26 )
    {
      v38 = 0;
      v42 = v44;
      v39 = 0;
      v40 = 0;
      v41 = 0;
      v43 = 0x400000000LL;
      goto LABEL_16;
    }
    v8 = *(_QWORD *)(a2 - 64);
  }
  v9 = *(_QWORD *)(a2 + 16);
  v38 = 0;
  v42 = v44;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v43 = 0x400000000LL;
  if ( v9 )
  {
    if ( (*(_BYTE *)(a2 + 1) & 1) == 0 || (sub_BD7FF0(a2, v8), (v9 = *(_QWORD *)(a2 + 16)) != 0) )
    {
      do
      {
        v10 = *(unsigned __int8 **)(v9 + 24);
        v11 = *v10;
        if ( v11 == 26 )
        {
          *((_DWORD *)v10 + 20) = -1;
        }
        else if ( v11 == 27 )
        {
          *((_DWORD *)v10 + 21) = -1;
          sub_AC2B30((__int64)(v10 - 32), 0);
        }
        if ( v6 && **(_BYTE **)(v9 + 24) == 28 )
        {
          v45 = *(_BYTE **)(v9 + 24);
          sub_D6CE50((__int64)&v38, (__int64 *)&v45, a3, a4, a5, a6);
        }
        sub_AC2B30(v9, v8);
        v9 = *(_QWORD *)(a2 + 16);
      }
      while ( v9 );
    }
  }
LABEL_16:
  sub_103E3E0(*a1, a2);
  sub_103CDC0(*a1, a2, 1);
  if ( !(_DWORD)v43 )
    goto LABEL_17;
  v15 = 8LL * (unsigned int)v43;
  v16 = v42;
  v45 = v47;
  v17 = v47;
  v46 = 0x1000000000LL;
  v18 = &v42[v15];
  v19 = v15 >> 3;
  if ( (unsigned __int64)v15 > 0x80 )
  {
    a2 = v15 >> 3;
    v35 = v42;
    sub_D6B130((__int64)&v45, v15 >> 3, v15, 0x1000000000LL, v12, v13);
    v16 = v35;
    v17 = &v45[24 * (unsigned int)v46];
  }
  do
  {
    if ( v17 )
    {
      v20 = *v16;
      *v17 = 4;
      v17[1] = 0;
      v17[2] = v20;
      LOBYTE(a2) = v20 != 0;
      if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
      {
        v34 = v16;
        sub_BD73F0((__int64)v17);
        v16 = v34;
      }
    }
    ++v16;
    v17 += 3;
  }
  while ( v18 != v16 );
  ++v38;
  LODWORD(v46) = v46 + v19;
  LODWORD(v21) = v46;
  if ( !(_DWORD)v40 )
  {
    if ( !HIDWORD(v40) )
      goto LABEL_35;
    a2 = (unsigned int)v41;
    if ( (unsigned int)v41 <= 0x40 )
      goto LABEL_31;
    a2 = 8LL * (unsigned int)v41;
    sub_C7D6A0((__int64)v39, a2, 8);
    LODWORD(v41) = 0;
    goto LABEL_51;
  }
  v22 = 4 * v40;
  a2 = (unsigned int)v41;
  if ( (unsigned int)(4 * v40) < 0x40 )
    v22 = 64;
  if ( v22 < (unsigned int)v41 )
  {
    if ( (_DWORD)v40 == 1 )
    {
      v32 = 86;
    }
    else
    {
      _BitScanReverse(&v30, v40 - 1);
      v31 = 1 << (33 - (v30 ^ 0x1F));
      if ( v31 < 64 )
        v31 = 64;
      if ( (_DWORD)v41 == v31 )
        goto LABEL_62;
      v32 = 4 * v31 / 3u + 1;
    }
    a2 = 8LL * (unsigned int)v41;
    sub_C7D6A0((__int64)v39, a2, 8);
    v33 = sub_AF1560(v32);
    LODWORD(v41) = v33;
    if ( v33 )
    {
      a2 = 8;
      v39 = (_QWORD *)sub_C7D670(8LL * v33, 8);
LABEL_62:
      sub_D6B710((__int64)&v38);
      LODWORD(v21) = v46;
      goto LABEL_35;
    }
LABEL_51:
    v39 = 0;
    LODWORD(v21) = v46;
    v40 = 0;
    goto LABEL_35;
  }
LABEL_31:
  v23 = v39;
  v24 = &v39[a2];
  if ( v39 != v24 )
  {
    do
      *v23++ = -4096;
    while ( v24 != v23 );
    LODWORD(v21) = v46;
  }
  v40 = 0;
LABEL_35:
  LODWORD(v43) = 0;
  v25 = v21 - 1;
  if ( (_DWORD)v21 )
  {
    do
    {
      while ( 1 )
      {
        v26 = (__int64)v45;
        v36[0] = 4;
        v36[1] = 0;
        a2 = (__int64)&v45[24 * (unsigned int)v21 - 24];
        v37 = *(_QWORD *)(a2 + 16);
        if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
        {
          a2 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(v36, a2);
          LODWORD(v21) = v46;
          v26 = (__int64)v45;
        }
        LODWORD(v46) = v21 - 1;
        sub_D68D70((_QWORD *)(v26 + 24LL * (unsigned int)(v21 - 1)));
        v27 = v37;
        if ( !v37 )
          break;
        sub_D68D70(v36);
        --v25;
        a2 = v27;
        sub_D6D630((__int64)a1, v27);
        v21 = (unsigned int)v46;
        if ( v25 == -1 )
          goto LABEL_43;
      }
      sub_D68D70(v36);
      --v25;
      v21 = (unsigned int)v46;
    }
    while ( v25 != -1 );
LABEL_43:
    v28 = v45;
    v29 = &v45[24 * v21];
    if ( v45 == (_BYTE *)v29 )
      goto LABEL_46;
    do
    {
      v29 -= 3;
      sub_D68D70(v29);
    }
    while ( v28 != (_BYTE *)v29 );
  }
  v28 = v45;
LABEL_46:
  if ( v28 != v47 )
    _libc_free(v28, a2);
LABEL_17:
  if ( v42 != v44 )
    _libc_free(v42, a2);
  return sub_C7D6A0((__int64)v39, 8LL * (unsigned int)v41, 8);
}
