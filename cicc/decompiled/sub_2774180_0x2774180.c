// Function: sub_2774180
// Address: 0x2774180
//
__int64 __fastcall sub_2774180(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned int v6; // edi
  __int64 result; // rax
  int v8; // ecx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // r14
  char v13; // al
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rax
  __int64 v17; // r14
  char i; // al
  _QWORD *v19; // r15
  __int64 v20; // rsi
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // r12
  __int64 v30; // r12
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // edx
  __int64 v36; // [rsp+0h] [rbp-E0h]
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __int64 v38; // [rsp+8h] [rbp-D8h]
  __int64 v39; // [rsp+18h] [rbp-C8h]
  _QWORD *v40; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v41; // [rsp+30h] [rbp-B0h]
  _QWORD v42[2]; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v43; // [rsp+48h] [rbp-98h]
  _QWORD v44[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v45; // [rsp+60h] [rbp-80h]
  __int64 v46; // [rsp+70h] [rbp-70h]
  __int64 v47; // [rsp+78h] [rbp-68h] BYREF
  __int64 v48; // [rsp+80h] [rbp-60h]
  __int64 v49; // [rsp+88h] [rbp-58h]
  __int64 v50; // [rsp+90h] [rbp-50h] BYREF
  __int64 v51; // [rsp+98h] [rbp-48h]
  __int64 v52; // [rsp+A0h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v39 = v4;
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
  result = sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  v9 = result;
  if ( v4 )
  {
    v10 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v36 = v5 << 6;
    v10 <<= 6;
    v11 = (v5 << 6) + v4;
    v46 = 0;
    v12 = result + v10;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    if ( result != result + v10 )
    {
      do
      {
        if ( v9 )
        {
          v13 = v46;
          *(_QWORD *)(v9 + 8) = 0;
          *(_QWORD *)(v9 + 16) = 0;
          *(_BYTE *)v9 = v13;
          v14 = v49;
          v15 = v49 == 0;
          *(_QWORD *)(v9 + 24) = v49;
          if ( v14 != -4096 && !v15 && v14 != -8192 )
            sub_BD6050((unsigned __int64 *)(v9 + 8), v47 & 0xFFFFFFFFFFFFFFF8LL);
          v16 = v52;
          *(_QWORD *)(v9 + 32) = 0;
          *(_QWORD *)(v9 + 40) = 0;
          *(_QWORD *)(v9 + 48) = v16;
          if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
            sub_BD6050((unsigned __int64 *)(v9 + 32), v50 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v9 += 64;
      }
      while ( v12 != v9 );
      if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
        sub_BD60C0(&v50);
      if ( v49 != -8192 && v49 != 0 && v49 != -4096 )
        sub_BD60C0(&v47);
    }
    v41 = 0;
    v42[0] = 0;
    v42[1] = 0;
    v43 = 0;
    v44[0] = 0;
    v44[1] = 0;
    v45 = 0;
    v46 = 1;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    if ( v11 != v4 )
    {
      v17 = v4;
      for ( i = 0; ; i = v41 )
      {
        v28 = *(_BYTE *)v17;
        if ( *(_BYTE *)v17 == i && *(_QWORD *)(v17 + 24) == v43 )
        {
          v26 = v45;
          if ( *(_QWORD *)(v17 + 48) == v45 )
            goto LABEL_36;
          if ( v28 == (_BYTE)v46 )
          {
LABEL_48:
            if ( *(_QWORD *)(v17 + 24) == v49 )
            {
              v26 = v52;
              if ( *(_QWORD *)(v17 + 48) == v52 )
                goto LABEL_36;
            }
          }
        }
        else if ( v28 == (_BYTE)v46 )
        {
          goto LABEL_48;
        }
        sub_2773FE0(a1, (char *)v17, &v40);
        v19 = v40;
        v20 = v40[3];
        *(_BYTE *)v40 = *(_BYTE *)v17;
        v21 = (__int64)(v19 + 1);
        v22 = *(_QWORD *)(v17 + 24);
        if ( v22 != v20 )
        {
          if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
          {
            v37 = *(_QWORD *)(v17 + 24);
            sub_BD60C0(v19 + 1);
            v22 = v37;
            v21 = (__int64)(v19 + 1);
          }
          v19[3] = v22;
          if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
            sub_BD73F0(v21);
        }
        v23 = *(_QWORD *)(v17 + 48);
        v24 = v19[6];
        v25 = (__int64)(v19 + 4);
        if ( v23 != v24 )
        {
          if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
          {
            v38 = *(_QWORD *)(v17 + 48);
            sub_BD60C0(v19 + 4);
            v23 = v38;
            v25 = (__int64)(v19 + 4);
          }
          v19[6] = v23;
          if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
            sub_BD73F0(v25);
        }
        v40[7] = *(_QWORD *)(v17 + 56);
        ++*(_DWORD *)(a1 + 16);
        v26 = *(_QWORD *)(v17 + 48);
LABEL_36:
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          sub_BD60C0((_QWORD *)(v17 + 32));
        v27 = *(_QWORD *)(v17 + 24);
        if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
          sub_BD60C0((_QWORD *)(v17 + 8));
        v17 += 64;
        if ( v11 == v17 )
        {
          if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
            sub_BD60C0(&v50);
          if ( v49 != -8192 && v49 != -4096 && v49 != 0 )
            sub_BD60C0(&v47);
          break;
        }
      }
    }
    if ( v45 != 0 && v45 != -4096 && v45 != -8192 )
      sub_BD60C0(v44);
    if ( v43 != 0 && v43 != -4096 && v43 != -8192 )
      sub_BD60C0(v42);
    return sub_C7D6A0(v39, v36, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v29 = *(unsigned int *)(a1 + 24);
    v46 = 0;
    v47 = 0;
    v30 = result + (v29 << 6);
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    if ( result != v30 )
    {
      do
      {
        if ( v9 )
        {
          v31 = v46;
          *(_QWORD *)(v9 + 8) = 0;
          *(_QWORD *)(v9 + 16) = 0;
          *(_BYTE *)v9 = v31;
          v32 = v49;
          v15 = v49 == -4096;
          *(_QWORD *)(v9 + 24) = v49;
          if ( v32 != 0 && !v15 && v32 != -8192 )
            sub_BD6050((unsigned __int64 *)(v9 + 8), v47 & 0xFFFFFFFFFFFFFFF8LL);
          v33 = v52;
          *(_QWORD *)(v9 + 32) = 0;
          *(_QWORD *)(v9 + 40) = 0;
          *(_QWORD *)(v9 + 48) = v33;
          if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
            sub_BD6050((unsigned __int64 *)(v9 + 32), v50 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v9 += 64;
      }
      while ( v30 != v9 );
      LODWORD(v34) = v52;
      if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
        v34 = sub_BD60C0(&v50);
      v35 = v49;
      LOBYTE(v34) = v49 != -8192;
      LOBYTE(v8) = v49 != 0;
      LOBYTE(v35) = v49 != -4096;
      result = v35 & v8 & (unsigned int)v34;
      if ( (_BYTE)result )
        return sub_BD60C0(&v47);
    }
  }
  return result;
}
