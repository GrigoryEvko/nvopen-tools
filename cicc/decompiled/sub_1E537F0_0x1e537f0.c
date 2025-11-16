// Function: sub_1E537F0
// Address: 0x1e537f0
//
bool __fastcall sub_1E537F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // r9
  __int64 *v12; // rsi
  __int64 v13; // r14
  __int64 *v14; // r13
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // edx
  int v20; // r10d
  unsigned int v21; // edx
  __int64 *v22; // rbx
  __int64 *v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 *v27; // r12
  __int64 *v28; // rdx
  int v29; // esi
  unsigned int v30; // eax
  __int64 v31; // rdi
  int v32; // eax
  int v33; // esi
  unsigned int v34; // eax
  __int64 v35; // rdi
  int v36; // eax
  __int64 *v38; // [rsp+8h] [rbp-58h]
  __int64 *v39; // [rsp+10h] [rbp-50h]
  __int64 *v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+28h] [rbp-38h] BYREF

  v4 = a2;
  v5 = a1;
  sub_1E48140(a2);
  *(_DWORD *)(a2 + 88) = 0;
  v38 = *(__int64 **)(a1 + 40);
  if ( *(__int64 **)(a1 + 32) != v38 )
  {
    v39 = *(__int64 **)(a1 + 32);
    v8 = (__int64 *)&v41;
    while ( 1 )
    {
      v9 = *v39;
      v10 = *(__int64 **)(*v39 + 112);
      v11 = (__int64)&v10[2 * *(unsigned int *)(*v39 + 120)];
      if ( v10 == (__int64 *)v11 )
        goto LABEL_16;
      v12 = v8;
      v13 = v4;
      v14 = &v10[2 * *(unsigned int *)(*v39 + 120)];
      do
      {
        while ( 1 )
        {
          v18 = *v10;
          if ( a3 )
          {
            v19 = *(_DWORD *)(a3 + 24);
            if ( !v19 )
              goto LABEL_7;
            v6 = (unsigned int)(v19 - 1);
            v7 = *(_QWORD *)(a3 + 8);
            v20 = 1;
            v21 = v6 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v11 = *(_QWORD *)(v7 + 8LL * v21);
            if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) != v11 )
            {
              while ( v11 != -8 )
              {
                v21 = v6 & (v20 + v21);
                v11 = *(_QWORD *)(v7 + 8LL * v21);
                if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) == v11 )
                  goto LABEL_11;
                ++v20;
              }
              goto LABEL_7;
            }
          }
LABEL_11:
          if ( (((unsigned __int8)v18 ^ 6) & 6) != 0 )
          {
            v15 = *(unsigned int *)(v5 + 24);
            v16 = v18 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !(_DWORD)v15 )
              goto LABEL_14;
          }
          else
          {
            if ( *((_DWORD *)v10 + 2) == 3 )
              goto LABEL_7;
            v15 = *(unsigned int *)(v5 + 24);
            v16 = v18 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !(_DWORD)v15 )
              goto LABEL_14;
          }
          v6 = (unsigned int)(v15 - 1);
          v7 = *(_QWORD *)(v5 + 8);
          v15 = (unsigned int)v6 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v17 = *(_QWORD *)(v7 + 8 * v15);
          if ( v16 != v17 )
            break;
LABEL_7:
          v10 += 2;
          if ( v10 == v14 )
            goto LABEL_15;
        }
        v11 = 1;
        while ( v17 != -8 )
        {
          v15 = (unsigned int)v6 & ((_DWORD)v11 + (_DWORD)v15);
          v17 = *(_QWORD *)(v7 + 8LL * (unsigned int)v15);
          if ( v16 == v17 )
            goto LABEL_7;
          v11 = (unsigned int)(v11 + 1);
        }
LABEL_14:
        v10 += 2;
        v41 = v16;
        sub_1E532F0(v13, v12, v15, v6, v7, (__int64 *)v11);
      }
      while ( v10 != v14 );
LABEL_15:
      v4 = v13;
      v8 = v12;
      v9 = *v39;
LABEL_16:
      v22 = *(__int64 **)(v9 + 32);
      v23 = &v22[2 * *(unsigned int *)(v9 + 40)];
      if ( v22 == v23 )
        goto LABEL_27;
      v24 = v8;
      v25 = v4;
      v26 = v5;
      v27 = v23;
      v28 = v24;
      while ( 2 )
      {
        while ( 2 )
        {
          v6 = *v22;
          if ( ((*v22 >> 1) & 3) != 1 )
            goto LABEL_19;
          v6 &= 0xFFFFFFFFFFFFFFF8LL;
          if ( !a3 )
            break;
          v32 = *(_DWORD *)(a3 + 24);
          if ( !v32 )
            goto LABEL_19;
          v33 = v32 - 1;
          v7 = *(_QWORD *)(a3 + 8);
          v11 = 1;
          v34 = (v32 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v35 = *(_QWORD *)(v7 + 8LL * v34);
          if ( v35 != v6 )
          {
            while ( v35 != -8 )
            {
              v34 = v33 & (v11 + v34);
              v35 = *(_QWORD *)(v7 + 8LL * v34);
              if ( v35 == v6 )
                goto LABEL_24;
              v11 = (unsigned int)(v11 + 1);
            }
LABEL_19:
            v22 += 2;
            if ( v27 == v22 )
              goto LABEL_26;
            continue;
          }
          break;
        }
LABEL_24:
        v36 = *(_DWORD *)(v26 + 24);
        if ( !v36 )
          goto LABEL_25;
        v29 = v36 - 1;
        v7 = *(_QWORD *)(v26 + 8);
        v30 = (v36 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v31 = *(_QWORD *)(v7 + 8LL * v30);
        if ( v31 == v6 )
          goto LABEL_19;
        v11 = 1;
        while ( v31 != -8 )
        {
          v30 = v29 & (v11 + v30);
          v31 = *(_QWORD *)(v7 + 8LL * v30);
          if ( v31 == v6 )
            goto LABEL_19;
          v11 = (unsigned int)(v11 + 1);
        }
LABEL_25:
        v22 += 2;
        v40 = v28;
        v41 = v6;
        sub_1E532F0(v25, v28, (__int64)v28, v6, v7, (__int64 *)v11);
        v28 = v40;
        if ( v27 != v22 )
          continue;
        break;
      }
LABEL_26:
      v5 = v26;
      v4 = v25;
      v8 = v28;
LABEL_27:
      if ( v38 == ++v39 )
        return *(_DWORD *)(v4 + 88) != 0;
    }
  }
  return 0;
}
