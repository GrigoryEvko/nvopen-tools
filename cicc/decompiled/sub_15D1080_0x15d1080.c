// Function: sub_15D1080
// Address: 0x15d1080
//
void __fastcall sub_15D1080(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r15
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r13
  const __m128i *v7; // r12
  __int64 v8; // rbx
  unsigned int v9; // ecx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  char v12; // al
  int v13; // eax
  __int64 v14; // rcx
  int v15; // esi
  int v16; // r10d
  __int64 *v17; // rdx
  unsigned int i; // r9d
  __int64 v19; // rdi
  __int64 v20; // r8
  unsigned int v21; // r9d
  unsigned int v22; // esi
  int v23; // r14d
  __int64 *v24; // rcx
  unsigned __int64 v25; // rdx
  char v26; // di
  __int64 v27; // r9
  int v28; // esi
  unsigned int v29; // r10d
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r8
  int v32; // eax
  __int64 *v33; // r8
  unsigned int j; // eax
  __int64 v35; // r10
  __int64 *v36; // r11
  unsigned int v37; // eax
  unsigned int v38; // esi
  unsigned int v39; // ecx
  int v40; // edi
  unsigned int v41; // r8d
  __int64 v42; // rax
  unsigned int v43; // ecx
  int v44; // edx
  unsigned int v45; // eax
  __int64 v46; // rax
  __int64 *v47; // [rsp+0h] [rbp-A0h]
  __int64 v48; // [rsp+8h] [rbp-98h]
  int v51; // [rsp+28h] [rbp-78h]
  int v52; // [rsp+2Ch] [rbp-74h]
  __int64 *v53; // [rsp+30h] [rbp-70h]
  __int64 v54; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v55; // [rsp+48h] [rbp-58h] BYREF
  __int64 v56; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v57; // [rsp+58h] [rbp-48h]
  __int64 *v58; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v59; // [rsp+68h] [rbp-38h]

  v54 = a3;
  if ( a1 == a2 || a2 == a1 + 2 )
    return;
  v3 = a1 + 2;
  do
  {
    while ( sub_15D0FA0(&v54, v3, a1) )
    {
      v4 = *v3;
      v5 = v3[1];
      if ( a1 != v3 )
        memmove(a1 + 2, a1, (char *)v3 - (char *)a1);
      v3 += 2;
      *a1 = v4;
      a1[1] = v5;
      if ( a2 == v3 )
        return;
    }
    v6 = *v3;
    v7 = (const __m128i *)v3;
    v47 = v3;
    v8 = v54;
    v48 = v3[1];
    v9 = (unsigned int)v48 >> 9;
    v10 = (((v9 ^ ((unsigned int)v48 >> 4)
           | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v9 ^ ((unsigned int)v48 >> 4)) << 32)) >> 22)
        ^ ((v9 ^ ((unsigned int)v48 >> 4) | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v9 ^ ((unsigned int)v48 >> 4)) << 32));
    v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
        ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
    v52 = ((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - ((_DWORD)v11 << 27));
    while ( 2 )
    {
      v57 = v48 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = *(_BYTE *)(v8 + 8);
      v53 = (__int64 *)v7;
      v56 = v6;
      v13 = v12 & 1;
      if ( v13 )
      {
        v14 = v8 + 16;
        v15 = 3;
      }
      else
      {
        v22 = *(_DWORD *)(v8 + 24);
        v14 = *(_QWORD *)(v8 + 16);
        if ( !v22 )
        {
          v39 = *(_DWORD *)(v8 + 8);
          ++*(_QWORD *)v8;
          v17 = 0;
          v40 = (v39 >> 1) + 1;
          goto LABEL_39;
        }
        v15 = v22 - 1;
      }
      v16 = 1;
      v17 = 0;
      for ( i = v15 & v52; ; i = v15 & v21 )
      {
        v19 = v14 + 24LL * i;
        v20 = *(_QWORD *)v19;
        if ( v6 == *(_QWORD *)v19 && *(_QWORD *)(v19 + 8) == (v48 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v23 = *(_DWORD *)(v19 + 16);
          goto LABEL_23;
        }
        if ( v20 == -8 )
          break;
        if ( v20 == -16 && *(_QWORD *)(v19 + 8) == -16 && !v17 )
          v17 = (__int64 *)(v14 + 24LL * i);
LABEL_18:
        v21 = v16 + i;
        ++v16;
      }
      if ( *(_QWORD *)(v19 + 8) != -8 )
        goto LABEL_18;
      v39 = *(_DWORD *)(v8 + 8);
      v41 = 12;
      v22 = 4;
      if ( !v17 )
        v17 = (__int64 *)v19;
      ++*(_QWORD *)v8;
      v40 = (v39 >> 1) + 1;
      if ( !(_BYTE)v13 )
      {
        v22 = *(_DWORD *)(v8 + 24);
LABEL_39:
        v41 = 3 * v22;
      }
      if ( v41 <= 4 * v40 )
      {
        v22 *= 2;
      }
      else if ( v22 - *(_DWORD *)(v8 + 12) - v40 > v22 >> 3 )
      {
        goto LABEL_42;
      }
      sub_15D0B40(v8, v22);
      sub_15D0A10(v8, &v56, &v58);
      v17 = v58;
      v39 = *(_DWORD *)(v8 + 8);
LABEL_42:
      *(_DWORD *)(v8 + 8) = (2 * (v39 >> 1) + 2) | v39 & 1;
      if ( *v17 != -8 || v17[1] != -8 )
        --*(_DWORD *)(v8 + 12);
      v42 = v56;
      *((_DWORD *)v17 + 4) = 0;
      v23 = 0;
      *v17 = v42;
      v17[1] = v57;
LABEL_23:
      v24 = (__int64 *)v7[-1].m128i_i64[0];
      v25 = v7[-1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v58 = v24;
      v59 = v25;
      v26 = *(_BYTE *)(v8 + 8) & 1;
      if ( v26 )
      {
        v27 = v8 + 16;
        v28 = 3;
LABEL_25:
        v51 = 1;
        v29 = (unsigned int)v25 >> 9;
        v30 = (((v29 ^ ((unsigned int)v25 >> 4)
               | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))
              - 1
              - ((unsigned __int64)(v29 ^ ((unsigned int)v25 >> 4)) << 32)) >> 22)
            ^ ((v29 ^ ((unsigned int)v25 >> 4)
              | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))
             - 1
             - ((unsigned __int64)(v29 ^ ((unsigned int)v25 >> 4)) << 32));
        v31 = 9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13)));
        v32 = (((v31 ^ (v31 >> 15)) - 1 - ((v31 ^ (v31 >> 15)) << 27)) >> 31)
            ^ ((v31 ^ (v31 >> 15)) - 1 - (((unsigned int)v31 ^ (unsigned int)(v31 >> 15)) << 27));
        v33 = 0;
        for ( j = v28 & v32; ; j = v28 & v37 )
        {
          v35 = v27 + 24LL * j;
          v36 = *(__int64 **)v35;
          if ( v24 == *(__int64 **)v35 && v25 == *(_QWORD *)(v35 + 8) )
          {
            --v7;
            if ( *(_DWORD *)(v35 + 16) >= v23 )
              goto LABEL_52;
            goto LABEL_37;
          }
          if ( v36 == (__int64 *)-8LL )
          {
            if ( *(_QWORD *)(v35 + 8) == -8 )
            {
              v43 = *(_DWORD *)(v8 + 8);
              v45 = 12;
              v38 = 4;
              if ( !v33 )
                v33 = (__int64 *)v35;
              ++*(_QWORD *)v8;
              v44 = (v43 >> 1) + 1;
              if ( !v26 )
              {
                v38 = *(_DWORD *)(v8 + 24);
                goto LABEL_46;
              }
              goto LABEL_47;
            }
          }
          else if ( v36 == (__int64 *)-16LL && *(_QWORD *)(v35 + 8) == -16 && !v33 )
          {
            v33 = (__int64 *)(v27 + 24LL * j);
          }
          v37 = v51 + j;
          ++v51;
        }
      }
      v38 = *(_DWORD *)(v8 + 24);
      v27 = *(_QWORD *)(v8 + 16);
      if ( v38 )
      {
        v28 = v38 - 1;
        goto LABEL_25;
      }
      v43 = *(_DWORD *)(v8 + 8);
      ++*(_QWORD *)v8;
      v33 = 0;
      v44 = (v43 >> 1) + 1;
LABEL_46:
      v45 = 3 * v38;
LABEL_47:
      if ( v45 <= 4 * v44 )
      {
        v38 *= 2;
LABEL_61:
        sub_15D0B40(v8, v38);
        sub_15D0A10(v8, (__int64 *)&v58, &v55);
        v33 = v55;
        v43 = *(_DWORD *)(v8 + 8);
        goto LABEL_49;
      }
      if ( v38 - *(_DWORD *)(v8 + 12) - v44 <= v38 >> 3 )
        goto LABEL_61;
LABEL_49:
      *(_DWORD *)(v8 + 8) = (2 * (v43 >> 1) + 2) | v43 & 1;
      if ( *v33 != -8 || v33[1] != -8 )
        --*(_DWORD *)(v8 + 12);
      --v7;
      *v33 = (__int64)v58;
      v46 = v59;
      *((_DWORD *)v33 + 4) = 0;
      v33[1] = v46;
      if ( v23 > 0 )
      {
LABEL_37:
        v7[1] = _mm_loadu_si128(v7);
        continue;
      }
      break;
    }
LABEL_52:
    *v53 = v6;
    v3 += 2;
    v53[1] = v48;
  }
  while ( a2 != v47 + 2 );
}
