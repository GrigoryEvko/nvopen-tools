// Function: sub_EE2F30
// Address: 0xee2f30
//
unsigned __int64 *__fastcall sub_EE2F30(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned __int16 *v5; // r12
  __int64 v6; // rax
  __int64 v7; // r15
  bool v8; // zf
  _WORD *v9; // r13
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 *v16; // rdi
  __int64 v17; // rcx
  int v18; // r14d
  __int64 *v19; // r10
  int v20; // ecx
  int v21; // ecx
  int v23; // esi
  int v24; // esi
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rdi
  int v28; // r11d
  __int64 *v29; // r9
  int v30; // esi
  int v31; // esi
  __int64 v32; // r8
  int v33; // r11d
  __int64 v34; // rdx
  __int64 v35; // rdi
  unsigned __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+28h] [rbp-58h]
  unsigned __int64 v43; // [rsp+30h] [rbp-50h] BYREF
  __m128i v44; // [rsp+38h] [rbp-48h] BYREF

  sub_C2DFC0(&v43, a2 + 16, *(__int64 **)(a2 + 8));
  v2 = v43 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v2 | 1;
  }
  else
  {
    v3 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8LL);
    v4 = *(_QWORD *)(v3 + 8);
    v5 = *(unsigned __int16 **)(v3 + 72);
    v6 = 10;
    v42 = v4;
    if ( v4 )
    {
      v7 = a2 + 16;
      while ( 1 )
      {
        v9 = sub_EE20B0((__int64)v5 + v6 + 16, *(_QWORD *)((char *)v5 + v6));
        v11 = v10;
        v12 = sub_EF75F0(v7, v9, v10);
        if ( !v12 )
          goto LABEL_4;
        v44.m128i_i64[0] = (__int64)v9;
        v44.m128i_i64[1] = v11;
        v13 = *(_DWORD *)(a2 + 48);
        v39 = a2 + 24;
        if ( v13 )
        {
          v14 = *(_QWORD *)(a2 + 32);
          v15 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * (_DWORD)v12)) & (v13 - 1);
          v16 = (__int64 *)(v14 + 24 * v15);
          v17 = *v16;
          if ( v12 == *v16 )
            goto LABEL_4;
          v18 = 1;
          v19 = 0;
          while ( v17 != -1 )
          {
            if ( v19 || v17 != -2 )
              v16 = v19;
            LODWORD(v15) = (v13 - 1) & (v18 + v15);
            v17 = *(_QWORD *)(v14 + 24LL * (unsigned int)v15);
            if ( v12 == v17 )
              goto LABEL_4;
            ++v18;
            v19 = v16;
            v16 = (__int64 *)(v14 + 24LL * (unsigned int)v15);
          }
          if ( !v19 )
            v19 = v16;
          v20 = *(_DWORD *)(a2 + 40);
          ++*(_QWORD *)(a2 + 24);
          v21 = v20 + 1;
          if ( 4 * v21 < 3 * v13 )
          {
            if ( v13 - *(_DWORD *)(a2 + 44) - v21 > v13 >> 3 )
              goto LABEL_17;
            v38 = v12;
            v36 = ((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (0xBF58476D1CE4E5B9LL * v12);
            sub_9E2150(v39, v13);
            v30 = *(_DWORD *)(a2 + 48);
            if ( !v30 )
            {
LABEL_49:
              ++*(_DWORD *)(a2 + 40);
              BUG();
            }
            v31 = v30 - 1;
            v32 = *(_QWORD *)(a2 + 32);
            v29 = 0;
            v33 = 1;
            LODWORD(v34) = v31 & v36;
            v19 = (__int64 *)(v32 + 24LL * (v31 & (unsigned int)v36));
            v21 = *(_DWORD *)(a2 + 40) + 1;
            v12 = v38;
            v35 = *v19;
            if ( v38 == *v19 )
              goto LABEL_17;
            while ( v35 != -1 )
            {
              if ( !v29 && v35 == -2 )
                v29 = v19;
              v34 = v31 & (unsigned int)(v34 + v33);
              v19 = (__int64 *)(v32 + 24 * v34);
              v35 = *v19;
              if ( v38 == *v19 )
                goto LABEL_17;
              ++v33;
            }
            goto LABEL_28;
          }
        }
        else
        {
          ++*(_QWORD *)(a2 + 24);
        }
        v37 = v12;
        sub_9E2150(v39, 2 * v13);
        v23 = *(_DWORD *)(a2 + 48);
        if ( !v23 )
          goto LABEL_49;
        v12 = v37;
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a2 + 32);
        LODWORD(v26) = v24 & (((0xBF58476D1CE4E5B9LL * v37) >> 31) ^ (484763065 * v37));
        v19 = (__int64 *)(v25 + 24LL * (unsigned int)v26);
        v27 = *v19;
        v21 = *(_DWORD *)(a2 + 40) + 1;
        if ( v37 == *v19 )
          goto LABEL_17;
        v28 = 1;
        v29 = 0;
        while ( v27 != -1 )
        {
          if ( v27 == -2 && !v29 )
            v29 = v19;
          v26 = v24 & (unsigned int)(v26 + v28);
          v19 = (__int64 *)(v25 + 24 * v26);
          v27 = *v19;
          if ( v37 == *v19 )
            goto LABEL_17;
          ++v28;
        }
LABEL_28:
        if ( v29 )
          v19 = v29;
LABEL_17:
        *(_DWORD *)(a2 + 40) = v21;
        if ( *v19 != -1 )
          --*(_DWORD *)(a2 + 44);
        *v19 = v12;
        *(__m128i *)(v19 + 1) = _mm_loadu_si128(&v44);
LABEL_4:
        if ( !v2 )
          v2 = *v5++;
        --v2;
        v8 = v42-- == 1;
        v5 = (unsigned __int16 *)((char *)v5 + *((_QWORD *)v5 + 1) + *((_QWORD *)v5 + 2) + 24);
        if ( v8 )
          break;
        v6 = v2 == 0 ? 10LL : 8LL;
      }
    }
    *a1 = 1;
  }
  return a1;
}
