// Function: sub_35540D0
// Address: 0x35540d0
//
bool __fastcall sub_35540D0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r13
  unsigned __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rdi
  int v17; // esi
  int v18; // r9d
  unsigned int v19; // edx
  __int64 v20; // r8
  int v21; // eax
  __int64 v22; // rdi
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 *v27; // r12
  __int64 v28; // r14
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // rsi
  int v32; // ecx
  int v33; // r8d
  unsigned int v34; // eax
  __int64 v35; // rdi
  int v36; // eax
  __int64 v37; // rsi
  int v38; // ecx
  int v39; // r8d
  unsigned int v40; // eax
  __int64 v41; // rdi
  int v43; // r9d
  __int64 *v44; // [rsp+8h] [rbp-68h]
  __int64 *v47; // [rsp+28h] [rbp-48h]
  __int64 v48[7]; // [rsp+38h] [rbp-38h] BYREF

  sub_35480E0(a2);
  *(_DWORD *)(a2 + 40) = 0;
  v6 = *(__int64 **)(a1 + 32);
  v44 = &v6[*(unsigned int *)(a1 + 40)];
  if ( v6 != v44 )
  {
    v47 = *(__int64 **)(a1 + 32);
    while ( 1 )
    {
      v7 = *v47;
      v8 = sub_35459D0(a3, *v47);
      v9 = *(_QWORD *)v8;
      v10 = *(_QWORD *)v8 + 32LL * *(unsigned int *)(v8 + 8);
      if ( *(_QWORD *)v8 == v10 )
        goto LABEL_13;
      v11 = a4;
      v12 = v9;
      v13 = v11;
      do
      {
        v14 = *(_QWORD *)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        v48[0] = v14;
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
          if ( (unsigned __int8)sub_3545640(v12, 1u) )
            goto LABEL_11;
          v21 = *(_DWORD *)(a1 + 24);
          v22 = *(_QWORD *)(a1 + 8);
          if ( v21 )
          {
            v23 = v21 - 1;
            v24 = (v21 - 1) & ((LODWORD(v48[0]) >> 9) ^ (LODWORD(v48[0]) >> 4));
            v25 = *(_QWORD *)(v22 + 8LL * v24);
            if ( v48[0] == v25 )
              goto LABEL_11;
            v43 = 1;
            while ( v25 != -4096 )
            {
              v24 = v23 & (v43 + v24);
              v25 = *(_QWORD *)(v22 + 8LL * v24);
              if ( v48[0] == v25 )
                goto LABEL_11;
              ++v43;
            }
          }
          sub_3553D90(a2, v48);
        }
LABEL_11:
        v12 += 32;
      }
      while ( v10 != v12 );
      a4 = v13;
LABEL_13:
      v26 = sub_3545E90(a3, v7);
      v27 = *(__int64 **)v26;
      v28 = *(_QWORD *)v26 + 32LL * *(unsigned int *)(v26 + 8);
      if ( v28 != *(_QWORD *)v26 )
      {
        while ( 2 )
        {
          v29 = *v27;
          v48[0] = *v27;
          if ( ((v27[1] >> 1) & 3) == 1 )
          {
            if ( !a4 )
              goto LABEL_18;
            v30 = *(_DWORD *)(a4 + 24);
            v31 = *(_QWORD *)(a4 + 8);
            if ( v30 )
            {
              v32 = v30 - 1;
              v33 = 1;
              v34 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
              v35 = *(_QWORD *)(v31 + 8LL * v34);
              if ( v29 == v35 )
              {
LABEL_18:
                v36 = *(_DWORD *)(a1 + 24);
                v37 = *(_QWORD *)(a1 + 8);
                if ( v36 )
                {
                  v38 = v36 - 1;
                  v39 = 1;
                  v40 = (v36 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                  v41 = *(_QWORD *)(v37 + 8LL * v40);
                  if ( v29 == v41 )
                    goto LABEL_20;
                  while ( v41 != -4096 )
                  {
                    v40 = v38 & (v39 + v40);
                    v41 = *(_QWORD *)(v37 + 8LL * v40);
                    if ( v29 == v41 )
                      goto LABEL_20;
                    ++v39;
                  }
                }
                sub_3553D90(a2, v48);
              }
              else
              {
                while ( v35 != -4096 )
                {
                  v34 = v32 & (v33 + v34);
                  v35 = *(_QWORD *)(v31 + 8LL * v34);
                  if ( v29 == v35 )
                    goto LABEL_18;
                  ++v33;
                }
              }
            }
          }
LABEL_20:
          v27 += 4;
          if ( (__int64 *)v28 == v27 )
            break;
          continue;
        }
      }
      if ( v44 == ++v47 )
        return *(_DWORD *)(a2 + 40) != 0;
    }
  }
  return 0;
}
