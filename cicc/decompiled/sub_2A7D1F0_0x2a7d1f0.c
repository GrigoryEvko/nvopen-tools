// Function: sub_2A7D1F0
// Address: 0x2a7d1f0
//
__int64 __fastcall sub_2A7D1F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  unsigned __int64 *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 i; // rax
  _QWORD *v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // r9
  _QWORD *v24; // rax
  int v25; // edx
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  bool v32; // al
  bool v33; // zf
  __int64 v35; // [rsp+0h] [rbp-C0h]
  __int64 v36; // [rsp+8h] [rbp-B8h]
  __int64 v37; // [rsp+10h] [rbp-B0h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  _QWORD *v39; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v40[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v41; // [rsp+40h] [rbp-80h]
  _QWORD v42[2]; // [rsp+48h] [rbp-78h] BYREF
  __int64 v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h] BYREF
  __int64 v45; // [rsp+68h] [rbp-58h]
  __int64 v46; // [rsp+70h] [rbp-50h]
  __int64 v47; // [rsp+78h] [rbp-48h] BYREF
  __int64 v48; // [rsp+80h] [rbp-40h]
  __int64 v49; // [rsp+88h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v38 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670(80LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( !v5 )
    return sub_2A7D090(a1, 8, v8, v9);
  v10 = (unsigned __int64 *)v7;
  v11 = *(unsigned int *)(a1 + 24);
  v44 = 0;
  v35 = 80 * v4;
  v12 = 80 * v4 + v5;
  v45 = 0;
  v13 = &v10[10 * v11];
  v46 = -4096;
  *(_QWORD *)(a1 + 16) = 0;
  v47 = 0;
  v48 = 0;
  v49 = -4096;
  if ( v10 != v13 )
  {
    do
    {
      if ( v10 )
      {
        v14 = v46;
        *v10 = 0;
        v10[1] = 0;
        v10[2] = v14;
        if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
          sub_BD6050(v10, v44 & 0xFFFFFFFFFFFFFFF8LL);
        v15 = v49;
        v10[3] = 0;
        v10[4] = 0;
        v10[5] = v15;
        if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
          sub_BD6050(v10 + 3, v47 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v10 += 10;
    }
    while ( v13 != v10 );
    if ( v49 != -4096 && v49 != 0 && v49 != -8192 )
      sub_BD60C0(&v47);
    if ( v46 != 0 && v46 != -4096 && v46 != -8192 )
      sub_BD60C0(&v44);
  }
  v41 = -4096;
  v40[0] = 0;
  v40[1] = 0;
  v42[0] = 0;
  v42[1] = 0;
  v43 = -4096;
  v44 = 0;
  v45 = 0;
  v46 = -8192;
  v47 = 0;
  v48 = 0;
  v49 = -8192;
  if ( v12 != v5 )
  {
    v16 = v5;
    for ( i = -4096; ; i = v41 )
    {
      v30 = *(_QWORD *)(v16 + 16);
      if ( v30 == i )
      {
        v28 = *(_QWORD *)(v16 + 40);
        if ( v43 == v28 )
          goto LABEL_42;
        if ( v30 == v46 )
        {
LABEL_53:
          v28 = *(_QWORD *)(v16 + 40);
          if ( v49 == v28 )
            goto LABEL_42;
        }
      }
      else if ( v30 == v46 )
      {
        goto LABEL_53;
      }
      sub_2A75400(a1, v16, &v39);
      v18 = v39;
      v19 = *(_QWORD *)(v16 + 16);
      v20 = v39[2];
      if ( v19 != v20 )
      {
        if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
        {
          v37 = *(_QWORD *)(v16 + 16);
          sub_BD60C0(v39);
          v19 = v37;
        }
        v18[2] = v19;
        if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          sub_BD73F0((__int64)v18);
      }
      v21 = *(_QWORD *)(v16 + 40);
      v22 = v18[5];
      v23 = (__int64)(v18 + 3);
      if ( v21 != v22 )
      {
        if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
        {
          v36 = *(_QWORD *)(v16 + 40);
          sub_BD60C0(v18 + 3);
          v21 = v36;
          v23 = (__int64)(v18 + 3);
        }
        v18[5] = v21;
        if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
          sub_BD73F0(v23);
      }
      v24 = v39;
      *((_DWORD *)v39 + 14) = *(_DWORD *)(v16 + 56);
      v24[6] = *(_QWORD *)(v16 + 48);
      v25 = *(_DWORD *)(v16 + 72);
      *(_DWORD *)(v16 + 56) = 0;
      *((_DWORD *)v24 + 18) = v25;
      v24[8] = *(_QWORD *)(v16 + 64);
      *(_DWORD *)(v16 + 72) = 0;
      ++*(_DWORD *)(a1 + 16);
      if ( *(_DWORD *)(v16 + 72) > 0x40u )
      {
        v26 = *(_QWORD *)(v16 + 64);
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
      if ( *(_DWORD *)(v16 + 56) > 0x40u )
      {
        v27 = *(_QWORD *)(v16 + 48);
        if ( v27 )
          j_j___libc_free_0_0(v27);
      }
      v28 = *(_QWORD *)(v16 + 40);
LABEL_42:
      if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
        sub_BD60C0((_QWORD *)(v16 + 24));
      v29 = *(_QWORD *)(v16 + 16);
      if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
        sub_BD60C0((_QWORD *)v16);
      v16 += 80;
      if ( v12 == v16 )
      {
        if ( v49 == 0 || v49 == -4096 || v49 == -8192 )
        {
          v31 = v46;
          v32 = v46 != -4096;
          v33 = v46 == 0;
        }
        else
        {
          sub_BD60C0(&v47);
          v31 = v46;
          v32 = v46 != 0;
          v33 = v46 == -4096;
        }
        if ( v31 != -8192 && !v33 && v32 )
          sub_BD60C0(&v44);
        break;
      }
    }
  }
  if ( v43 != -4096 && v43 != 0 && v43 != -8192 )
    sub_BD60C0(v42);
  if ( v41 != 0 && v41 != -4096 && v41 != -8192 )
    sub_BD60C0(v40);
  return sub_C7D6A0(v38, v35, 8);
}
