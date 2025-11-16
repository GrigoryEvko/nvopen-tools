// Function: sub_35543E0
// Address: 0x35543e0
//
bool __fastcall sub_35543E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 *v9; // rcx
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdi
  int v17; // esi
  int v18; // r9d
  unsigned int v19; // eax
  __int64 v20; // r8
  int v21; // eax
  __int64 v22; // rdi
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // rsi
  int v34; // ecx
  int v35; // r8d
  unsigned int v36; // eax
  __int64 v37; // rdi
  int v38; // eax
  __int64 v39; // rsi
  int v40; // ecx
  int v41; // r8d
  unsigned int v42; // eax
  __int64 v43; // rdi
  int v45; // r9d
  __int64 *v46; // [rsp+8h] [rbp-68h]
  __int64 *v49; // [rsp+28h] [rbp-48h]
  __int64 v50[7]; // [rsp+38h] [rbp-38h] BYREF

  sub_35480E0(a2);
  *(_DWORD *)(a2 + 40) = 0;
  v6 = *(__int64 **)(a1 + 32);
  v46 = &v6[*(unsigned int *)(a1 + 40)];
  if ( v6 != v46 )
  {
    v49 = *(__int64 **)(a1 + 32);
    while ( 1 )
    {
      v7 = *v49;
      v8 = sub_3545E90(a3, *v49);
      v9 = *(__int64 **)v8;
      v10 = *(_QWORD *)v8 + 32LL * *(unsigned int *)(v8 + 8);
      if ( *(_QWORD *)v8 == v10 )
        goto LABEL_13;
      v11 = a4;
      v12 = v9;
      v13 = v11;
      do
      {
        v14 = *v12;
        v50[0] = *v12;
        if ( !v13 )
          goto LABEL_8;
        v15 = *(_DWORD *)(v13 + 24);
        v16 = *(_QWORD *)(v13 + 8);
        if ( !v15 )
          goto LABEL_11;
        v17 = v15 - 1;
        v18 = 1;
        v19 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v20 = *(_QWORD *)(v16 + 8LL * v19);
        if ( v14 != v20 )
        {
          while ( v20 != -4096 )
          {
            v19 = v17 & (v18 + v19);
            v20 = *(_QWORD *)(v16 + 8LL * v19);
            if ( v14 == v20 )
              goto LABEL_8;
            ++v18;
          }
        }
        else
        {
LABEL_8:
          if ( (unsigned __int8)sub_3545640((__int64)v12, 0) )
            goto LABEL_11;
          v21 = *(_DWORD *)(a1 + 24);
          v22 = *(_QWORD *)(a1 + 8);
          if ( v21 )
          {
            v23 = v21 - 1;
            v24 = (v21 - 1) & ((LODWORD(v50[0]) >> 9) ^ (LODWORD(v50[0]) >> 4));
            v25 = *(_QWORD *)(v22 + 8LL * v24);
            if ( v50[0] == v25 )
              goto LABEL_11;
            v45 = 1;
            while ( v25 != -4096 )
            {
              v24 = v23 & (v45 + v24);
              v25 = *(_QWORD *)(v22 + 8LL * v24);
              if ( v50[0] == v25 )
                goto LABEL_11;
              ++v45;
            }
          }
          sub_3553D90(a2, v50);
        }
LABEL_11:
        v12 += 4;
      }
      while ( (__int64 *)v10 != v12 );
      a4 = v13;
LABEL_13:
      v26 = sub_35459D0(a3, v7);
      v27 = *(_QWORD *)v26;
      v28 = *(_QWORD *)v26 + 32LL * *(unsigned int *)(v26 + 8);
      if ( v28 != *(_QWORD *)v26 )
      {
        while ( 2 )
        {
          v29 = *(_QWORD *)(v27 + 8);
          v30 = v29 >> 1;
          v31 = v29 & 0xFFFFFFFFFFFFFFF8LL;
          v50[0] = v31;
          if ( (v30 & 3) == 1 )
          {
            if ( !a4 )
              goto LABEL_18;
            v32 = *(_DWORD *)(a4 + 24);
            v33 = *(_QWORD *)(a4 + 8);
            if ( v32 )
            {
              v34 = v32 - 1;
              v35 = 1;
              v36 = (v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
              v37 = *(_QWORD *)(v33 + 8LL * v36);
              if ( v31 == v37 )
              {
LABEL_18:
                v38 = *(_DWORD *)(a1 + 24);
                v39 = *(_QWORD *)(a1 + 8);
                if ( v38 )
                {
                  v40 = v38 - 1;
                  v41 = 1;
                  v42 = (v38 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
                  v43 = *(_QWORD *)(v39 + 8LL * v42);
                  if ( v31 == v43 )
                    goto LABEL_20;
                  while ( v43 != -4096 )
                  {
                    v42 = v40 & (v41 + v42);
                    v43 = *(_QWORD *)(v39 + 8LL * v42);
                    if ( v31 == v43 )
                      goto LABEL_20;
                    ++v41;
                  }
                }
                sub_3553D90(a2, v50);
              }
              else
              {
                while ( v37 != -4096 )
                {
                  v36 = v34 & (v35 + v36);
                  v37 = *(_QWORD *)(v33 + 8LL * v36);
                  if ( v31 == v37 )
                    goto LABEL_18;
                  ++v35;
                }
              }
            }
          }
LABEL_20:
          v27 += 32;
          if ( v27 == v28 )
            break;
          continue;
        }
      }
      if ( v46 == ++v49 )
        return *(_DWORD *)(a2 + 40) != 0;
    }
  }
  return 0;
}
