// Function: sub_24AE680
// Address: 0x24ae680
//
void __fastcall sub_24AE680(__int64 a1)
{
  __int64 v1; // r15
  unsigned __int64 v2; // rbx
  _BYTE *v3; // r12
  __int64 v4; // rcx
  __int64 v5; // rdi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r11
  __int64 v9; // r13
  int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned int v13; // eax
  __int64 v14; // r9
  unsigned int v15; // r8d
  __int64 v16; // rbx
  unsigned int v17; // r11d
  __int64 v18; // rax
  unsigned int v19; // ebx
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // rbx
  __int64 v25; // r12
  __int64 *v26; // r13
  __int64 v27; // rsi
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // r12
  char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rdx
  int v41; // r9d
  __int64 v42; // [rsp+8h] [rbp-1A8h]
  unsigned int v43; // [rsp+8h] [rbp-1A8h]
  unsigned int v44; // [rsp+10h] [rbp-1A0h]
  __int64 v45; // [rsp+10h] [rbp-1A0h]
  __int64 v47; // [rsp+28h] [rbp-188h]
  void *s; // [rsp+30h] [rbp-180h] BYREF
  __int64 v49; // [rsp+38h] [rbp-178h]
  _BYTE v50[16]; // [rsp+40h] [rbp-170h] BYREF
  __int64 v51[2]; // [rsp+50h] [rbp-160h] BYREF
  _BYTE v52[16]; // [rsp+60h] [rbp-150h] BYREF
  _QWORD v53[4]; // [rsp+70h] [rbp-140h] BYREF
  __m128i v54; // [rsp+90h] [rbp-120h] BYREF
  __int64 *v55; // [rsp+A0h] [rbp-110h]
  __int16 v56; // [rsp+B0h] [rbp-100h]
  __m128i v57[2]; // [rsp+C0h] [rbp-F0h] BYREF
  char v58; // [rsp+E0h] [rbp-D0h]
  char v59; // [rsp+E1h] [rbp-CFh]
  __m128i v60[3]; // [rsp+F0h] [rbp-C0h] BYREF
  __m128i v61[2]; // [rsp+120h] [rbp-90h] BYREF
  char v62; // [rsp+140h] [rbp-70h]
  char v63; // [rsp+141h] [rbp-6Fh]
  __m128i v64[6]; // [rsp+150h] [rbp-60h] BYREF

  v47 = *(_QWORD *)a1 + 72LL;
  if ( *(_QWORD *)(*(_QWORD *)a1 + 80LL) != v47 )
  {
    v1 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    while ( 1 )
    {
      if ( !v1 )
        BUG();
      v2 = *(_QWORD *)(v1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v1 + 24 == v2 )
      {
        v3 = 0;
      }
      else
      {
        if ( !v2 )
          BUG();
        v3 = (_BYTE *)(v2 - 24);
        if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 >= 0xB )
          v3 = 0;
      }
      if ( (unsigned int)sub_B46E30((__int64)v3) <= 1 || (unsigned __int8)(*v3 - 31) > 3u && *v3 != 40 )
        goto LABEL_3;
      v4 = *(unsigned int *)(a1 + 296);
      v5 = *(_QWORD *)(a1 + 280);
      if ( (_DWORD)v4 )
      {
        v6 = (v4 - 1) & (((unsigned int)(v1 - 24) >> 9) ^ ((unsigned int)(v1 - 24) >> 4));
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( v1 - 24 == *v7 )
          goto LABEL_14;
        v30 = 1;
        while ( v8 != -4096 )
        {
          v41 = v30 + 1;
          v6 = (v4 - 1) & (v30 + v6);
          v7 = (__int64 *)(v5 + 16LL * v6);
          v8 = *v7;
          if ( v1 - 24 == *v7 )
            goto LABEL_14;
          v30 = v41;
        }
      }
      v7 = (__int64 *)(v5 + 16 * v4);
LABEL_14:
      v9 = v7[1];
      if ( !*(_QWORD *)(v9 + 16) )
        goto LABEL_3;
      if ( v1 + 24 == v2 )
      {
        v11 = 0;
      }
      else
      {
        if ( !v2 )
          BUG();
        v10 = *(unsigned __int8 *)(v2 - 24);
        v11 = 0;
        v12 = v2 - 24;
        if ( (unsigned int)(v10 - 30) < 0xB )
          v11 = v12;
      }
      v44 = *(_DWORD *)(v9 + 80);
      v13 = sub_B46E30(v11);
      v15 = v44;
      v16 = v13;
      v17 = v13;
      s = v50;
      v49 = 0x200000000LL;
      if ( v13 > 2 )
      {
        v43 = v13;
        sub_C8D5F0((__int64)&s, v50, v13, 8u, v44, v14);
        memset(s, 0, 8 * v16);
        v15 = v44;
        LODWORD(v49) = v43;
      }
      else
      {
        if ( v13 )
        {
          v18 = 8 * v13;
          *(_QWORD *)&v50[v18 - 8] = 0;
          if ( (unsigned int)(v18 - 1) >= 8 )
          {
            v19 = (v18 - 1) & 0xFFFFFFF8;
            v20 = 0;
            do
            {
              v21 = v20;
              v20 += 8;
              *(_QWORD *)&v50[v21] = 0;
            }
            while ( v20 < v19 );
          }
        }
        LODWORD(v49) = v17;
      }
      if ( !v15 )
        goto LABEL_39;
      v45 = v1;
      v22 = 0;
      v23 = v9;
      v42 = (__int64)v3;
      v24 = 0;
      v25 = 8LL * v15;
      do
      {
        v26 = *(__int64 **)(*(_QWORD *)(v23 + 72) + v24);
        v27 = v26[1];
        if ( v27 )
        {
          v28 = sub_D0E820(*v26, v27);
          v29 = v26[4];
          *((_QWORD *)s + v28) = v29;
          if ( v22 < v29 )
            v22 = v29;
        }
        v24 += 8;
      }
      while ( v25 != v24 );
      v1 = v45;
      if ( v22 )
      {
        sub_24AD4F0(*(_QWORD *)(a1 + 8), v42, s, (unsigned int)v49, v22);
      }
      else
      {
LABEL_39:
        v31 = *(_QWORD *)a1;
        v32 = **(_QWORD **)(a1 + 8);
        v63 = 1;
        v61[0].m128i_i64[0] = (__int64)", possibly due to the lack of a return path.";
        v62 = 3;
        v59 = 1;
        v57[0].m128i_i64[0] = (__int64)" partially ignored";
        v58 = 3;
        v33 = (char *)sub_BD5D20(v31);
        if ( v33 )
        {
          v51[0] = (__int64)v52;
          sub_24A2F70(v51, v33, (__int64)&v33[v34]);
        }
        else
        {
          v52[0] = 0;
          v51[0] = (__int64)v52;
          v51[1] = 0;
        }
        v55 = v51;
        v54.m128i_i64[0] = (__int64)"Profile in ";
        v56 = 1027;
        sub_9C6370(v60, &v54, v57, v35, (__int64)v60, v36);
        sub_9C6370(v64, v60, v61, v37, (__int64)v60, v38);
        v53[3] = v64;
        v39 = *(_QWORD *)(a1 + 8);
        v53[1] = 0x100000017LL;
        v40 = *(_QWORD *)(v39 + 168);
        v53[0] = &unk_49D9CA8;
        v53[2] = v40;
        sub_B6EB20(v32, (__int64)v53);
        if ( (_BYTE *)v51[0] != v52 )
          j_j___libc_free_0(v51[0]);
      }
      if ( s == v50 )
      {
LABEL_3:
        v1 = *(_QWORD *)(v1 + 8);
        if ( v47 == v1 )
          return;
      }
      else
      {
        _libc_free((unsigned __int64)s);
        v1 = *(_QWORD *)(v1 + 8);
        if ( v47 == v1 )
          return;
      }
    }
  }
}
