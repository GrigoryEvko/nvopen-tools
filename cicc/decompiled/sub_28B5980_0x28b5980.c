// Function: sub_28B5980
// Address: 0x28b5980
//
void __fastcall sub_28B5980(__int64 *a1)
{
  __int64 *v1; // rbx
  unsigned int v2; // edx
  unsigned int v3; // ecx
  __int64 *v4; // r14
  __int64 *v5; // r15
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned int v8; // eax
  __int64 *v9; // r9
  __int64 *v10; // r8
  __int64 *v11; // rdx
  __int64 *v12; // rcx
  __int64 *v13; // rax
  unsigned int v14; // eax
  int v15; // esi
  __int64 *v16; // rax
  __int64 *v17; // rsi
  __int64 v18; // r10
  __int64 v19; // rax
  int v20; // edi
  bool v21; // cc
  unsigned __int64 v22; // rdi
  int v23; // eax
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rdx
  int v30; // eax
  unsigned int v31; // [rsp+0h] [rbp-100h]
  unsigned int v32; // [rsp+8h] [rbp-F8h]
  _BYTE v33[24]; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-D8h]
  unsigned int v35; // [rsp+30h] [rbp-D0h]
  unsigned int v36; // [rsp+80h] [rbp-80h]
  unsigned __int64 v37; // [rsp+88h] [rbp-78h] BYREF
  unsigned int v38; // [rsp+90h] [rbp-70h]
  unsigned int v39; // [rsp+A8h] [rbp-58h]
  unsigned __int64 v40; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v41; // [rsp+B8h] [rbp-48h]

  v1 = a1 + 11;
  sub_28B4EA0((__int64)v33, a1);
  while ( 1 )
  {
    v2 = v36;
    v3 = *((_DWORD *)v1 - 42);
    v4 = v1 - 11;
    v5 = v1 - 35;
    if ( v36 == v3 )
    {
      v31 = *((_DWORD *)v1 - 42);
      v32 = v36;
      v28 = sub_C4C880((__int64)&v37, (__int64)(v1 - 20));
      v2 = v32;
      v3 = v31;
      v6 = v28 >> 31;
    }
    else
    {
      LOBYTE(v6) = v36 < v3;
    }
    if ( !(_BYTE)v6 )
    {
      if ( v2 == v3 )
        v7 = (unsigned int)sub_C4C880((__int64)(v1 - 20), (__int64)&v37) >> 31;
      else
        LOBYTE(v7) = v2 > v3;
      if ( (_BYTE)v7 )
        break;
      v8 = *((_DWORD *)v1 - 32);
      if ( v39 == v8 )
        v8 = (unsigned int)sub_C4C880((__int64)&v40, (__int64)(v1 - 15)) >> 31;
      else
        LOBYTE(v8) = v39 < v8;
      if ( !(_BYTE)v8 )
        break;
    }
    v9 = v1 - 34;
    v10 = v1 - 10;
    *(v1 - 11) = *(v1 - 35);
    if ( (v4[2] & 1) == 0 )
    {
      sub_C7D6A0(*(v1 - 8), 8LL * *((unsigned int *)v1 - 14), 8);
      v10 = v1 - 10;
      v9 = v1 - 34;
    }
    *((_DWORD *)v4 + 4) = 1;
    v11 = v1 - 8;
    *((_DWORD *)v1 - 17) = 0;
    v12 = v1 - 8;
    v13 = v1 - 8;
    do
    {
      if ( v13 )
        *v13 = -4096;
      ++v13;
    }
    while ( v13 != v1 );
    v14 = v5[2] & 0xFFFFFFFE;
    *((_DWORD *)v5 + 4) = v4[2] & 0xFFFFFFFE | v5[2] & 1;
    *((_DWORD *)v4 + 4) = v14 | v4[2] & 1;
    v15 = *((_DWORD *)v1 - 65);
    *((_DWORD *)v1 - 65) = *((_DWORD *)v1 - 17);
    *((_DWORD *)v1 - 17) = v15;
    if ( (v4[2] & 1) == 0 )
    {
      if ( (v5[2] & 1) == 0 )
      {
        v29 = *(v1 - 32);
        *(v1 - 32) = *(v1 - 8);
        v30 = *((_DWORD *)v1 - 14);
        *(v1 - 8) = v29;
        LODWORD(v29) = *((_DWORD *)v1 - 62);
        *((_DWORD *)v1 - 62) = v30;
        *((_DWORD *)v1 - 14) = v29;
        goto LABEL_32;
      }
      v16 = v9;
      v17 = v1 - 8;
      v12 = v1 - 32;
      v9 = v10;
      v10 = v16;
LABEL_29:
      *((_BYTE *)v9 + 8) |= 1u;
      v18 = v9[2];
      v19 = 0;
      v20 = *((_DWORD *)v9 + 6);
      do
      {
        v17[v19] = v12[v19];
        ++v19;
      }
      while ( v19 != 8 );
      *((_BYTE *)v10 + 8) &= ~1u;
      v10[2] = v18;
      *((_DWORD *)v10 + 6) = v20;
      goto LABEL_32;
    }
    v26 = v1 - 32;
    v17 = v1 - 32;
    if ( (v5[2] & 1) == 0 )
      goto LABEL_29;
    do
    {
      v27 = *v11;
      *v11++ = *v26;
      *v26++ = v27;
    }
    while ( v1 != v11 );
LABEL_32:
    v21 = *((_DWORD *)v1 + 10) <= 0x40u;
    *(_BYTE *)v1 = *((_BYTE *)v1 - 192);
    *((_DWORD *)v1 + 1) = *((_DWORD *)v1 - 47);
    v1[1] = *(v1 - 23);
    v1[2] = *(v1 - 22);
    *((_DWORD *)v1 + 6) = *((_DWORD *)v1 - 42);
    if ( !v21 )
    {
      v22 = v1[4];
      if ( v22 )
        j_j___libc_free_0_0(v22);
    }
    v21 = *((_DWORD *)v1 + 20) <= 0x40u;
    v1[4] = *(v1 - 20);
    v23 = *((_DWORD *)v1 - 38);
    *((_DWORD *)v1 - 38) = 0;
    *((_DWORD *)v1 + 10) = v23;
    v1[6] = *(v1 - 18);
    v1[7] = *(v1 - 17);
    *((_DWORD *)v1 + 16) = *((_DWORD *)v1 - 32);
    if ( !v21 )
    {
      v24 = v1[9];
      if ( v24 )
        j_j___libc_free_0_0(v24);
    }
    v25 = *(v1 - 15);
    v1 -= 24;
    v1[33] = v25;
    LODWORD(v25) = *((_DWORD *)v1 + 20);
    *((_DWORD *)v1 + 20) = 0;
    *((_DWORD *)v1 + 68) = v25;
    *((_DWORD *)v1 + 70) = *((_DWORD *)v1 + 22);
    v1[36] = v1[12];
  }
  sub_28B56D0((__int64)(v1 - 11), (__int64)v33);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( (v33[16] & 1) == 0 )
    sub_C7D6A0(v34, 8LL * v35, 8);
}
