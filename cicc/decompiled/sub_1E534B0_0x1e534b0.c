// Function: sub_1E534B0
// Address: 0x1e534b0
//
bool __fastcall sub_1E534B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 *v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // r12
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  int v20; // edx
  int v21; // r10d
  unsigned int v22; // edx
  __int64 v23; // rdx
  __int64 *v24; // rbx
  __int64 *v25; // rdx
  __int64 *v26; // rax
  __int64 v27; // r14
  __int64 v28; // r13
  __int64 *v29; // r12
  __int64 *v30; // rdx
  int v31; // esi
  unsigned int v32; // eax
  __int64 v33; // rdi
  int v34; // eax
  int v35; // esi
  unsigned int v36; // eax
  __int64 v37; // rdi
  int v38; // eax
  __int64 *v40; // [rsp+8h] [rbp-58h]
  __int64 *v41; // [rsp+10h] [rbp-50h]
  __int64 *v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 v43; // [rsp+28h] [rbp-38h] BYREF

  v4 = a2;
  v5 = a1;
  sub_1E48140(a2);
  *(_DWORD *)(a2 + 88) = 0;
  v40 = *(__int64 **)(a1 + 40);
  if ( *(__int64 **)(a1 + 32) != v40 )
  {
    v42 = *(__int64 **)(a1 + 32);
    v8 = (__int64 *)&v43;
    while ( 1 )
    {
      v9 = *v42;
      v10 = *(_QWORD *)(*v42 + 32);
      v11 = v10 + 16LL * *(unsigned int *)(*v42 + 40);
      if ( v10 == v11 )
        goto LABEL_17;
      v12 = v8;
      v13 = v4;
      v14 = v5;
      v15 = v10 + 16LL * *(unsigned int *)(*v42 + 40);
      do
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)v10;
          if ( a3 )
          {
            v20 = *(_DWORD *)(a3 + 24);
            if ( !v20 )
              goto LABEL_8;
            v6 = (unsigned int)(v20 - 1);
            v11 = *(_QWORD *)(a3 + 8);
            v21 = 1;
            v22 = v6 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
            v7 = *(_QWORD *)(v11 + 8LL * v22);
            if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != v7 )
            {
              while ( v7 != -8 )
              {
                v22 = v6 & (v21 + v22);
                v7 = *(_QWORD *)(v11 + 8LL * v22);
                if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) == v7 )
                  goto LABEL_12;
                ++v21;
              }
              goto LABEL_8;
            }
          }
LABEL_12:
          v23 = (v19 >> 1) & 3;
          if ( v23 == 3 )
            break;
          if ( v23 != 1 )
          {
            v16 = *(unsigned int *)(v14 + 24);
            v17 = v19 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !(_DWORD)v16 )
              goto LABEL_15;
            goto LABEL_7;
          }
LABEL_8:
          v10 += 16;
          if ( v10 == v15 )
            goto LABEL_16;
        }
        if ( *(_DWORD *)(v10 + 8) == 3 )
          goto LABEL_8;
        v16 = *(unsigned int *)(v14 + 24);
        v17 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !(_DWORD)v16 )
          goto LABEL_15;
LABEL_7:
        v6 = (unsigned int)(v16 - 1);
        v11 = *(_QWORD *)(v14 + 8);
        v16 = (unsigned int)v6 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v18 = *(_QWORD *)(v11 + 8 * v16);
        if ( v17 == v18 )
          goto LABEL_8;
        v7 = 1;
        while ( v18 != -8 )
        {
          v16 = (unsigned int)v6 & ((_DWORD)v7 + (_DWORD)v16);
          v18 = *(_QWORD *)(v11 + 8LL * (unsigned int)v16);
          if ( v17 == v18 )
            goto LABEL_8;
          v7 = (unsigned int)(v7 + 1);
        }
LABEL_15:
        v10 += 16;
        v43 = v17;
        sub_1E532F0(v13, v12, v16, v6, v11, (__int64 *)v7);
      }
      while ( v10 != v15 );
LABEL_16:
      v5 = v14;
      v4 = v13;
      v8 = v12;
      v9 = *v42;
LABEL_17:
      v24 = *(__int64 **)(v9 + 112);
      v25 = &v24[2 * *(unsigned int *)(v9 + 120)];
      if ( v24 == v25 )
        goto LABEL_28;
      v26 = v8;
      v27 = v4;
      v28 = v5;
      v29 = v25;
      v30 = v26;
      while ( 2 )
      {
        while ( 2 )
        {
          v6 = *v24;
          if ( ((*v24 >> 1) & 3) != 1 )
            goto LABEL_20;
          v6 &= 0xFFFFFFFFFFFFFFF8LL;
          if ( !a3 )
            break;
          v34 = *(_DWORD *)(a3 + 24);
          if ( !v34 )
            goto LABEL_20;
          v35 = v34 - 1;
          v11 = *(_QWORD *)(a3 + 8);
          v7 = 1;
          v36 = (v34 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v37 = *(_QWORD *)(v11 + 8LL * v36);
          if ( v37 != v6 )
          {
            while ( v37 != -8 )
            {
              v36 = v35 & (v7 + v36);
              v37 = *(_QWORD *)(v11 + 8LL * v36);
              if ( v37 == v6 )
                goto LABEL_25;
              v7 = (unsigned int)(v7 + 1);
            }
LABEL_20:
            v24 += 2;
            if ( v29 == v24 )
              goto LABEL_27;
            continue;
          }
          break;
        }
LABEL_25:
        v38 = *(_DWORD *)(v28 + 24);
        if ( !v38 )
          goto LABEL_26;
        v31 = v38 - 1;
        v11 = *(_QWORD *)(v28 + 8);
        v32 = (v38 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v33 = *(_QWORD *)(v11 + 8LL * v32);
        if ( v33 == v6 )
          goto LABEL_20;
        v7 = 1;
        while ( v33 != -8 )
        {
          v32 = v31 & (v7 + v32);
          v33 = *(_QWORD *)(v11 + 8LL * v32);
          if ( v33 == v6 )
            goto LABEL_20;
          v7 = (unsigned int)(v7 + 1);
        }
LABEL_26:
        v24 += 2;
        v41 = v30;
        v43 = v6;
        sub_1E532F0(v27, v30, (__int64)v30, v6, v11, (__int64 *)v7);
        v30 = v41;
        if ( v29 != v24 )
          continue;
        break;
      }
LABEL_27:
      v5 = v28;
      v4 = v27;
      v8 = v30;
LABEL_28:
      if ( v40 == ++v42 )
        return *(_DWORD *)(v4 + 88) != 0;
    }
  }
  return 0;
}
