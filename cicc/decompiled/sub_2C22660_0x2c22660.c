// Function: sub_2C22660
// Address: 0x2c22660
//
void __fastcall sub_2C22660(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  char v6; // di
  int v7; // edi
  __int64 v8; // r8
  int v9; // esi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r15
  __int64 v14; // r14
  char v15; // dh
  __int64 v16; // rdi
  __int16 v17; // ax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r10
  unsigned __int64 v21; // rsi
  unsigned __int64 *v22; // rax
  int v23; // ecx
  unsigned __int64 *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r8
  unsigned int v36; // esi
  unsigned __int64 v37; // rdi
  unsigned int v38; // edx
  int v39; // ecx
  unsigned int v40; // r8d
  __int64 *v41; // rdx
  int v42; // r14d
  __int64 v43; // r10
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+10h] [rbp-A0h]
  __int64 v48; // [rsp+18h] [rbp-98h]
  __int64 v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  int v51; // [rsp+30h] [rbp-80h]
  char v52; // [rsp+37h] [rbp-79h]
  __int64 v53; // [rsp+38h] [rbp-78h]
  __int64 v55; // [rsp+70h] [rbp-40h] BYREF
  __int64 v56[7]; // [rsp+78h] [rbp-38h] BYREF

  v2 = 0;
  v49 = a2 + 120;
  v53 = *(_QWORD *)(a1 + 48);
  v50 = v53 + 8LL * *(unsigned int *)(a1 + 56);
  if ( v53 != v50 )
  {
    do
    {
      v4 = *(_QWORD *)(v53 + v2);
      if ( (unsigned __int8)sub_2AAA120(v4) )
      {
        v52 = 0;
        v51 = 0;
      }
      else
      {
        v51 = *(_DWORD *)(a2 + 8) - 1;
        v52 = *(_BYTE *)(a2 + 12);
      }
      v5 = sub_2BF0520(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 80) + 56LL) + v2));
      v6 = *(_BYTE *)(a2 + 128);
      v55 = v5;
      v7 = v6 & 1;
      if ( v7 )
      {
        v8 = a2 + 136;
        v9 = 3;
      }
      else
      {
        v36 = *(_DWORD *)(a2 + 144);
        v8 = *(_QWORD *)(a2 + 136);
        if ( !v36 )
        {
          v38 = *(_DWORD *)(a2 + 128);
          ++*(_QWORD *)(a2 + 120);
          v56[0] = 0;
          v39 = (v38 >> 1) + 1;
LABEL_46:
          v40 = 3 * v36;
          goto LABEL_47;
        }
        v9 = v36 - 1;
      }
      v10 = v9 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v5 == *v11 )
      {
LABEL_7:
        v13 = v11[1];
        goto LABEL_8;
      }
      v42 = 1;
      v43 = 0;
      while ( v12 != -4096 )
      {
        if ( !v43 && v12 == -8192 )
          v43 = (__int64)v11;
        v10 = v9 & (v42 + v10);
        v11 = (__int64 *)(v8 + 16LL * v10);
        v12 = *v11;
        if ( v5 == *v11 )
          goto LABEL_7;
        ++v42;
      }
      v40 = 12;
      v36 = 4;
      if ( !v43 )
        v43 = (__int64)v11;
      v38 = *(_DWORD *)(a2 + 128);
      ++*(_QWORD *)(a2 + 120);
      v56[0] = v43;
      v39 = (v38 >> 1) + 1;
      if ( !(_BYTE)v7 )
      {
        v36 = *(_DWORD *)(a2 + 144);
        goto LABEL_46;
      }
LABEL_47:
      if ( v40 <= 4 * v39 )
      {
        v36 *= 2;
LABEL_54:
        sub_2ACA3E0(v49, v36);
        sub_2ABFB80(v49, &v55, v56);
        v5 = v55;
        v38 = *(_DWORD *)(a2 + 128);
        goto LABEL_49;
      }
      if ( v36 - *(_DWORD *)(a2 + 132) - v39 <= v36 >> 3 )
        goto LABEL_54;
LABEL_49:
      *(_DWORD *)(a2 + 128) = (2 * (v38 >> 1) + 2) | v38 & 1;
      v41 = (__int64 *)v56[0];
      if ( *(_QWORD *)v56[0] != -4096 )
        --*(_DWORD *)(a2 + 132);
      *v41 = v5;
      v13 = 0;
      v41[1] = 0;
LABEL_8:
      v14 = *(_QWORD *)(a2 + 904);
      v16 = sub_AA4FF0(v13);
      if ( !v16 )
      {
        *(_QWORD *)(v14 + 48) = v13;
        *(_QWORD *)(v14 + 56) = 0;
        *(_WORD *)(v14 + 64) = 1;
LABEL_11:
        v56[0] = *(_QWORD *)sub_B46C60(v16);
        if ( v56[0] && (sub_2AAAFA0(v56), (v20 = v56[0]) != 0) )
        {
          v21 = *(unsigned int *)(v14 + 8);
          v22 = *(unsigned __int64 **)v14;
          v23 = *(_DWORD *)(v14 + 8);
          v24 = (unsigned __int64 *)(*(_QWORD *)v14 + 16 * v21);
          if ( *(unsigned __int64 **)v14 == v24 )
          {
LABEL_41:
            v37 = *(unsigned int *)(v14 + 12);
            if ( v21 >= v37 )
            {
              v44 = v21 + 1;
              v45 = v48 & 0xFFFFFFFF00000000LL;
              v48 &= 0xFFFFFFFF00000000LL;
              if ( v37 < v44 )
              {
                v46 = v45;
                v47 = v56[0];
                sub_C8D5F0(v14, (const void *)(v14 + 16), v44, 0x10u, v18, v19);
                v45 = v46;
                v20 = v47;
                v24 = (unsigned __int64 *)(*(_QWORD *)v14 + 16LL * *(unsigned int *)(v14 + 8));
              }
              *v24 = v45;
              v24[1] = v20;
              ++*(_DWORD *)(v14 + 8);
            }
            else
            {
              if ( v24 )
              {
                *(_DWORD *)v24 = 0;
                v24[1] = v20;
                v23 = *(_DWORD *)(v14 + 8);
              }
              *(_DWORD *)(v14 + 8) = v23 + 1;
            }
          }
          else
          {
            while ( *(_DWORD *)v22 )
            {
              v22 += 2;
              if ( v24 == v22 )
                goto LABEL_41;
            }
            v22[1] = v56[0];
          }
        }
        else
        {
          sub_93FB40(v14, 0);
        }
        sub_9C6650(v56);
        goto LABEL_19;
      }
      LOBYTE(v17) = 1;
      *(_QWORD *)(v14 + 48) = v13;
      HIBYTE(v17) = v15;
      *(_QWORD *)(v14 + 56) = v16;
      *(_WORD *)(v14 + 64) = v17;
      if ( v16 != v13 + 48 )
      {
        v16 -= 24;
        goto LABEL_11;
      }
LABEL_19:
      LODWORD(v56[0]) = v51;
      BYTE4(v56[0]) = v52;
      v25 = sub_2BFB120(a2, v4, (unsigned int *)v56);
      v26 = *(_QWORD *)(a1 + 96);
      v27 = *(_DWORD *)(v26 + 4) & 0x7FFFFFF;
      if ( v27 )
      {
        v28 = *(unsigned int *)(v26 + 72);
        v29 = *(_QWORD *)(v26 - 8);
        v30 = 0;
        do
        {
          if ( *(_QWORD *)(v29 + 32LL * *(unsigned int *)(v26 + 72) + 8 * v30) == v13 )
          {
            v31 = 0;
            v32 = 8LL * v27;
            while ( 1 )
            {
              if ( *(_QWORD *)(v29 + 32 * v28 + v31) != v13 )
                goto LABEL_24;
              v33 = v29 + 4 * v31;
              if ( *(_QWORD *)v33 )
              {
                v34 = *(_QWORD *)(v33 + 8);
                **(_QWORD **)(v33 + 16) = v34;
                if ( v34 )
                  *(_QWORD *)(v34 + 16) = *(_QWORD *)(v33 + 16);
              }
              *(_QWORD *)v33 = v25;
              if ( v25 )
              {
                v35 = *(_QWORD *)(v25 + 16);
                *(_QWORD *)(v33 + 8) = v35;
                if ( v35 )
                  *(_QWORD *)(v35 + 16) = v33 + 8;
                v31 += 8;
                *(_QWORD *)(v33 + 16) = v25 + 16;
                *(_QWORD *)(v25 + 16) = v33;
                if ( v32 == v31 )
                  goto LABEL_34;
              }
              else
              {
LABEL_24:
                v31 += 8;
                if ( v32 == v31 )
                  goto LABEL_34;
              }
              v29 = *(_QWORD *)(v26 - 8);
              v28 = *(unsigned int *)(v26 + 72);
            }
          }
          ++v30;
        }
        while ( v27 != (_DWORD)v30 );
      }
      sub_F0A850(v26, v25, v13);
LABEL_34:
      v2 += 8;
    }
    while ( v50 != v2 + v53 );
  }
  sub_A88F30(
    *(_QWORD *)(a2 + 904),
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL),
    0);
}
