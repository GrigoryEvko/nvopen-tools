// Function: sub_318EF00
// Address: 0x318ef00
//
__int64 __fastcall sub_318EF00(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // r13
  __int64 v4; // r14
  unsigned int v5; // eax
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // r15
  char i; // al
  _QWORD *v18; // r14
  __int64 v19; // rsi
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 v27; // dl
  __int64 result; // rax
  __int64 v29; // rax
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
  __int64 v40; // [rsp+28h] [rbp-B8h] BYREF
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
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v39 = v4;
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = sub_C7D670(72LL * v5, 8);
  *(_QWORD *)(a1 + 8) = v6;
  v8 = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v46 = 0;
    v36 = 72 * v3;
    v9 = 72 * v3 + v4;
    v10 = *(unsigned int *)(a1 + 24);
    v47 = 0;
    v48 = 0;
    v11 = v8 + 72 * v10;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    if ( v8 != v11 )
    {
      do
      {
        if ( v8 )
        {
          v12 = v46;
          *(_QWORD *)(v8 + 8) = 0;
          *(_QWORD *)(v8 + 16) = 0;
          *(_BYTE *)v8 = v12;
          v13 = v49;
          v14 = v49 == 0;
          *(_QWORD *)(v8 + 24) = v49;
          if ( v13 != -4096 && !v14 && v13 != -8192 )
            sub_BD6050((unsigned __int64 *)(v8 + 8), v47 & 0xFFFFFFFFFFFFFFF8LL);
          v15 = v52;
          *(_QWORD *)(v8 + 32) = 0;
          *(_QWORD *)(v8 + 40) = 0;
          *(_QWORD *)(v8 + 48) = v15;
          if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
            sub_BD6050((unsigned __int64 *)(v8 + 32), v50 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v8 += 72;
      }
      while ( v11 != v8 );
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
    if ( v9 != v4 )
    {
      v16 = v4;
      for ( i = 0; ; i = v41 )
      {
        v27 = *(_BYTE *)v16;
        if ( *(_BYTE *)v16 == i && *(_QWORD *)(v16 + 24) == v43 )
        {
          v25 = v45;
          if ( *(_QWORD *)(v16 + 48) == v45 )
            goto LABEL_36;
          if ( v27 == (_BYTE)v46 )
          {
LABEL_48:
            if ( *(_QWORD *)(v16 + 24) == v49 )
            {
              v25 = v52;
              if ( *(_QWORD *)(v16 + 48) == v52 )
                goto LABEL_36;
            }
          }
        }
        else if ( v27 == (_BYTE)v46 )
        {
          goto LABEL_48;
        }
        sub_318EE30(a1, (unsigned __int8 *)v16, &v40);
        v18 = (_QWORD *)v40;
        v19 = *(_QWORD *)(v40 + 24);
        *(_BYTE *)v40 = *(_BYTE *)v16;
        v20 = (__int64)(v18 + 1);
        v21 = *(_QWORD *)(v16 + 24);
        if ( v21 != v19 )
        {
          if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
          {
            v37 = *(_QWORD *)(v16 + 24);
            sub_BD60C0(v18 + 1);
            v21 = v37;
            v20 = (__int64)(v18 + 1);
          }
          v18[3] = v21;
          if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
            sub_BD73F0(v20);
        }
        v22 = *(_QWORD *)(v16 + 48);
        v23 = v18[6];
        v24 = (__int64)(v18 + 4);
        if ( v22 != v23 )
        {
          if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
          {
            v38 = *(_QWORD *)(v16 + 48);
            sub_BD60C0(v18 + 4);
            v22 = v38;
            v24 = (__int64)(v18 + 4);
          }
          v18[6] = v22;
          if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
            sub_BD73F0(v24);
        }
        *(__m128i *)(v40 + 56) = _mm_loadu_si128((const __m128i *)(v16 + 56));
        ++*(_DWORD *)(a1 + 16);
        v25 = *(_QWORD *)(v16 + 48);
LABEL_36:
        if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
          sub_BD60C0((_QWORD *)(v16 + 32));
        v26 = *(_QWORD *)(v16 + 24);
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          sub_BD60C0((_QWORD *)(v16 + 8));
        v16 += 72;
        if ( v9 == v16 )
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
    v29 = *(unsigned int *)(a1 + 24);
    v46 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    result = 9 * v29;
    v47 = 0;
    v30 = v8 + 8 * result;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    if ( v8 != v30 )
    {
      do
      {
        if ( v8 )
        {
          v31 = v46;
          *(_QWORD *)(v8 + 8) = 0;
          *(_QWORD *)(v8 + 16) = 0;
          *(_BYTE *)v8 = v31;
          v32 = v49;
          v14 = v49 == -4096;
          *(_QWORD *)(v8 + 24) = v49;
          if ( v32 != 0 && !v14 && v32 != -8192 )
            sub_BD6050((unsigned __int64 *)(v8 + 8), v47 & 0xFFFFFFFFFFFFFFF8LL);
          v33 = v52;
          *(_QWORD *)(v8 + 32) = 0;
          *(_QWORD *)(v8 + 40) = 0;
          *(_QWORD *)(v8 + 48) = v33;
          if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
            sub_BD6050((unsigned __int64 *)(v8 + 32), v50 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v8 += 72;
      }
      while ( v30 != v8 );
      LODWORD(v34) = v52;
      if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
        v34 = sub_BD60C0(&v50);
      v35 = v49;
      LOBYTE(v34) = v49 != -8192;
      LOBYTE(v7) = v49 != 0;
      LOBYTE(v35) = v49 != -4096;
      result = v35 & v7 & (unsigned int)v34;
      if ( (_BYTE)result )
        return sub_BD60C0(&v47);
    }
  }
  return result;
}
