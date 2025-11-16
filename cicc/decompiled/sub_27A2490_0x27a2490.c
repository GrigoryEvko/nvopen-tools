// Function: sub_27A2490
// Address: 0x27a2490
//
__int64 __fastcall sub_27A2490(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v19; // r15
  __int64 result; // rax
  __int64 v21; // rax
  int v22; // esi
  int v23; // esi
  unsigned int v24; // edi
  __int64 v25; // rdx
  __int64 v26; // r10
  unsigned int v27; // edi
  unsigned int v28; // edx
  __int64 v29; // rax
  __int64 v30; // r10
  int v31; // esi
  int v32; // esi
  unsigned int v33; // edi
  __int64 v34; // rdx
  __int64 v35; // r10
  unsigned int v36; // r10d
  unsigned int v37; // edi
  __int64 v38; // rdx
  int v39; // edx
  int v40; // edx
  int v41; // eax
  int v42; // edx
  int v43; // edx
  unsigned int v44; // [rsp+4h] [rbp-5Ch]
  __int64 v45; // [rsp+8h] [rbp-58h]
  unsigned int v47; // [rsp+18h] [rbp-48h]
  char v48; // [rsp+1Fh] [rbp-41h]
  __int64 v49; // [rsp+20h] [rbp-40h]
  __int64 v50; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 248);
  v6 = *(_QWORD *)(v5 + 72);
  v7 = *(unsigned int *)(v5 + 88);
  if ( (_DWORD)v7 )
  {
    v10 = (v7 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v11 = (__int64 *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( a4 == *v11 )
    {
LABEL_3:
      if ( v11 != (__int64 *)(v6 + 16 * v7) )
      {
        v13 = v11[1];
        if ( v13 )
        {
          v14 = *(_QWORD *)(a3 + 72);
          v15 = a2;
          v16 = *(_QWORD *)(v13 + 8);
          v45 = v14;
          v50 = *(_QWORD *)(v14 + 40);
          v49 = *(_QWORD *)(a2 + 40);
          if ( v16 != v13 )
          {
            v48 = 0;
            v47 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
            v17 = a3;
            v44 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
            v19 = v17;
            do
            {
              if ( !v16 )
                BUG();
              if ( *(_BYTE *)(v16 - 32) == 26 )
              {
                v21 = *(_QWORD *)(v16 + 40);
                if ( a4 == v50 )
                {
                  v31 = *(_DWORD *)(a1 + 288);
                  a5 = *(_QWORD *)(a1 + 272);
                  if ( v31 )
                  {
                    v32 = v31 - 1;
                    v33 = v32 & v47;
                    v34 = a5 + 16LL * (v32 & v47);
                    v35 = *(_QWORD *)v34;
                    if ( v45 == *(_QWORD *)v34 )
                    {
LABEL_22:
                      v36 = *(_DWORD *)(v34 + 8);
                    }
                    else
                    {
                      v39 = 1;
                      while ( v35 != -4096 )
                      {
                        v15 = (unsigned int)(v39 + 1);
                        v33 = v32 & (v39 + v33);
                        v34 = a5 + 16LL * v33;
                        v35 = *(_QWORD *)v34;
                        if ( v45 == *(_QWORD *)v34 )
                          goto LABEL_22;
                        v39 = v15;
                      }
                      v36 = 0;
                    }
                    v37 = v32 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                    v38 = a5 + 16LL * v37;
                    v12 = *(_QWORD *)v38;
                    if ( v21 == *(_QWORD *)v38 )
                    {
LABEL_24:
                      if ( *(_DWORD *)(v38 + 8) > v36 )
                        return 0;
                    }
                    else
                    {
                      v40 = 1;
                      while ( v12 != -4096 )
                      {
                        v15 = (unsigned int)(v40 + 1);
                        v37 = v32 & (v40 + v37);
                        v38 = a5 + 16LL * v37;
                        v12 = *(_QWORD *)v38;
                        if ( v21 == *(_QWORD *)v38 )
                          goto LABEL_24;
                        v40 = v15;
                      }
                    }
                  }
                }
                if ( a4 != v49 || v48 )
                  goto LABEL_7;
                v22 = *(_DWORD *)(a1 + 288);
                a5 = *(_QWORD *)(a1 + 272);
                if ( !v22 )
                  goto LABEL_19;
                v23 = v22 - 1;
                v24 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                v25 = a5 + 16LL * v24;
                v26 = *(_QWORD *)v25;
                if ( v21 == *(_QWORD *)v25 )
                {
LABEL_16:
                  v27 = *(_DWORD *)(v25 + 8);
                }
                else
                {
                  v42 = 1;
                  while ( v26 != -4096 )
                  {
                    v15 = (unsigned int)(v42 + 1);
                    v24 = v23 & (v42 + v24);
                    v25 = a5 + 16LL * v24;
                    v26 = *(_QWORD *)v25;
                    if ( v21 == *(_QWORD *)v25 )
                      goto LABEL_16;
                    v42 = v15;
                  }
                  v27 = 0;
                }
                v28 = v23 & v44;
                v29 = a5 + 16LL * (v23 & v44);
                v30 = *(_QWORD *)v29;
                if ( a2 != *(_QWORD *)v29 )
                {
                  v41 = 1;
                  while ( v30 != -4096 )
                  {
                    v15 = (unsigned int)(v41 + 1);
                    v28 = v23 & (v41 + v28);
                    v29 = a5 + 16LL * v28;
                    v30 = *(_QWORD *)v29;
                    if ( a2 == *(_QWORD *)v29 )
                      goto LABEL_18;
                    v41 = v15;
                  }
LABEL_19:
                  v48 = 1;
LABEL_7:
                  result = sub_103BBC0(v19, v16 - 32, *(_QWORD **)(a1 + 232), v15, a5, v12);
                  if ( (_BYTE)result )
                    return result;
                  goto LABEL_8;
                }
LABEL_18:
                if ( v27 >= *(_DWORD *)(v29 + 8) )
                  goto LABEL_19;
              }
LABEL_8:
              v16 = *(_QWORD *)(v16 + 8);
            }
            while ( v13 != v16 );
          }
        }
      }
    }
    else
    {
      v43 = 1;
      while ( v12 != -4096 )
      {
        a5 = (unsigned int)(v43 + 1);
        v10 = (v7 - 1) & (v43 + v10);
        v11 = (__int64 *)(v6 + 16LL * v10);
        v12 = *v11;
        if ( a4 == *v11 )
          goto LABEL_3;
        v43 = a5;
      }
    }
  }
  return 0;
}
