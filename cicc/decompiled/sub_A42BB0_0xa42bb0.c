// Function: sub_A42BB0
// Address: 0xa42bb0
//
void __fastcall sub_A42BB0(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  __int64 v6; // r10
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r9
  __int64 v10; // r10
  unsigned int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // r8
  unsigned int *v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rbx
  __int64 *v17; // rbx
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // r10
  unsigned int v21; // edx
  int v22; // eax
  _QWORD *v23; // rdi
  __int64 v24; // r8
  __int64 v25; // rax
  int v26; // eax
  int v27; // ecx
  int v28; // ecx
  __int64 v29; // r10
  unsigned int v30; // edx
  __int64 v31; // r8
  _QWORD *v32; // r11
  __m128i *v34; // [rsp-68h] [rbp-68h]
  __m128i *v35; // [rsp-60h] [rbp-60h]
  __int64 v36; // [rsp-60h] [rbp-60h]
  int v37; // [rsp-60h] [rbp-60h]
  __int64 v38; // [rsp-60h] [rbp-60h]
  int v39; // [rsp-60h] [rbp-60h]
  int v40; // [rsp-60h] [rbp-60h]
  __int64 v41; // [rsp-58h] [rbp-58h] BYREF
  __int64 v42; // [rsp-50h] [rbp-50h]
  char *v43; // [rsp-48h] [rbp-48h]

  if ( a2 != a3 )
  {
    v3 = a2;
    v4 = a2 + 1;
    if ( a2 + 1 != a3 && !*(_BYTE *)(a1 + 320) )
    {
      v6 = *(_QWORD *)(a1 + 112);
      v7 = 16LL * a3;
      v8 = 16LL * a2;
      v34 = (__m128i *)(v7 + v6);
      v35 = (__m128i *)(v6 + v8);
      sub_A40D50(&v41, (__m128i *)(v6 + v8), (v7 - v8) >> 4);
      if ( v43 )
        sub_A428E0(v35, v34, v43, v42, a1);
      else
        sub_A3E4C0(v35, v34, a1);
      j_j___libc_free_0(v43, 16 * v42);
      sub_A40FA0(
        (__m128i *)(v8 + *(_QWORD *)(a1 + 112)),
        (__int64 *)(v7 + *(_QWORD *)(a1 + 112)),
        (unsigned __int8 (__fastcall *)(char *))sub_A3CF30);
      v9 = a1 + 80;
      while ( 1 )
      {
        v15 = *(_DWORD *)(a1 + 104);
        v16 = v3;
        v3 = v4;
        v17 = (__int64 *)(*(_QWORD *)(a1 + 112) + 16 * v16);
        if ( v15 )
        {
          v10 = *(_QWORD *)(a1 + 88);
          v11 = (v15 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
          v12 = (_QWORD *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( *v17 == *v12 )
          {
LABEL_9:
            v14 = (unsigned int *)(v12 + 1);
            goto LABEL_10;
          }
          v37 = 1;
          v23 = 0;
          while ( v13 != -4096 )
          {
            if ( v13 == -8192 && !v23 )
              v23 = v12;
            v11 = (v15 - 1) & (v37 + v11);
            v12 = (_QWORD *)(v10 + 16LL * v11);
            v13 = *v12;
            if ( *v17 == *v12 )
              goto LABEL_9;
            ++v37;
          }
          if ( !v23 )
            v23 = v12;
          v26 = *(_DWORD *)(a1 + 96);
          ++*(_QWORD *)(a1 + 80);
          v22 = v26 + 1;
          if ( 4 * v22 < 3 * v15 )
          {
            if ( v15 - *(_DWORD *)(a1 + 100) - v22 > v15 >> 3 )
              goto LABEL_16;
            v38 = v9;
            sub_A429D0(v9, v15);
            v27 = *(_DWORD *)(a1 + 104);
            if ( !v27 )
            {
LABEL_51:
              ++*(_DWORD *)(a1 + 96);
              BUG();
            }
            v28 = v27 - 1;
            v29 = *(_QWORD *)(a1 + 88);
            v9 = v38;
            v30 = v28 & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
            v22 = *(_DWORD *)(a1 + 96) + 1;
            v23 = (_QWORD *)(v29 + 16LL * v30);
            v31 = *v23;
            if ( *v17 == *v23 )
              goto LABEL_16;
            v39 = 1;
            v32 = 0;
            while ( v31 != -4096 )
            {
              if ( v31 == -8192 && !v32 )
                v32 = v23;
              v30 = v28 & (v39 + v30);
              v23 = (_QWORD *)(v29 + 16LL * v30);
              v31 = *v23;
              if ( *v17 == *v23 )
                goto LABEL_16;
              ++v39;
            }
            goto LABEL_30;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 80);
        }
        v36 = v9;
        sub_A429D0(v9, 2 * v15);
        v18 = *(_DWORD *)(a1 + 104);
        if ( !v18 )
          goto LABEL_51;
        v19 = v18 - 1;
        v20 = *(_QWORD *)(a1 + 88);
        v9 = v36;
        v21 = v19 & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
        v22 = *(_DWORD *)(a1 + 96) + 1;
        v23 = (_QWORD *)(v20 + 16LL * v21);
        v24 = *v23;
        if ( *v23 == *v17 )
          goto LABEL_16;
        v40 = 1;
        v32 = 0;
        while ( v24 != -4096 )
        {
          if ( !v32 && v24 == -8192 )
            v32 = v23;
          v21 = v19 & (v40 + v21);
          v23 = (_QWORD *)(v20 + 16LL * v21);
          v24 = *v23;
          if ( *v17 == *v23 )
            goto LABEL_16;
          ++v40;
        }
LABEL_30:
        if ( v32 )
          v23 = v32;
LABEL_16:
        *(_DWORD *)(a1 + 96) = v22;
        if ( *v23 != -4096 )
          --*(_DWORD *)(a1 + 100);
        v25 = *v17;
        *((_DWORD *)v23 + 2) = 0;
        *v23 = v25;
        v14 = (unsigned int *)(v23 + 1);
LABEL_10:
        *v14 = v4;
        if ( a3 == v4 )
          return;
        ++v4;
      }
    }
  }
}
