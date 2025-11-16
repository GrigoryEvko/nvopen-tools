// Function: sub_3514A50
// Address: 0x3514a50
//
char __fastcall sub_3514A50(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 *v9; // r15
  _QWORD *v10; // rdi
  __int64 v11; // rsi
  unsigned int v12; // esi
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // rdi
  __int64 **v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rbx
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rsi
  int v22; // edi
  _QWORD *v23; // rcx
  int v24; // eax
  int v25; // eax
  int v26; // esi
  int v27; // esi
  unsigned int v28; // edx
  __int64 v29; // rdi
  int v30; // r11d
  _QWORD *v31; // r10
  int v32; // esi
  int v33; // esi
  int v34; // r11d
  unsigned int v35; // edx
  __int64 v36; // rdi
  int v38; // [rsp+Ch] [rbp-54h]
  unsigned int v39; // [rsp+Ch] [rbp-54h]
  __int64 v42[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(__int64 **)(a3 + 112);
  v6 = *(unsigned int *)(a3 + 120);
  if ( v5 != &v5[v6] )
  {
    v9 = &v5[v6];
    do
    {
      v18 = *v5;
      v42[0] = *v5;
      if ( !a5 )
        goto LABEL_4;
      if ( *(_DWORD *)(a5 + 16) )
      {
        LODWORD(v6) = *(_DWORD *)(a5 + 24);
        v19 = *(_QWORD *)(a5 + 8);
        if ( (_DWORD)v6 )
        {
          v20 = v6 - 1;
          LODWORD(v6) = (v6 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v21 = *(_QWORD *)(v19 + 8LL * (unsigned int)v6);
          if ( v18 != v21 )
          {
            v22 = 1;
            while ( v21 != -4096 )
            {
              LODWORD(v6) = v20 & (v22 + v6);
              v21 = *(_QWORD *)(v19 + 8LL * (unsigned int)v6);
              if ( v18 == v21 )
                goto LABEL_4;
              ++v22;
            }
            goto LABEL_15;
          }
LABEL_4:
          v12 = *(_DWORD *)(a1 + 912);
          if ( v12 )
          {
            v13 = *(_QWORD *)(a1 + 896);
            v14 = (v12 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v6 = v13 + 16 * v14;
            v15 = *(_QWORD *)v6;
            if ( v18 == *(_QWORD *)v6 )
            {
LABEL_6:
              v16 = *(__int64 ***)(v6 + 8);
              LOBYTE(v6) = a2 == v16;
LABEL_7:
              if ( v18 != a4 && !(_BYTE)v6 )
              {
                LODWORD(v6) = *((_DWORD *)v16 + 14);
                if ( (_DWORD)v6 )
                {
                  LODWORD(v6) = v6 - 1;
                  *((_DWORD *)v16 + 14) = v6;
                  if ( !(_DWORD)v6 )
                  {
                    v17 = **v16;
                    if ( *(_BYTE *)(v17 + 216) )
                    {
                      v6 = *(unsigned int *)(a1 + 352);
                      if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 356) )
                      {
                        sub_C8D5F0(a1 + 344, (const void *)(a1 + 360), v6 + 1, 8u, v14, v13);
                        v6 = *(unsigned int *)(a1 + 352);
                      }
                      *(_QWORD *)(*(_QWORD *)(a1 + 344) + 8 * v6) = v17;
                      ++*(_DWORD *)(a1 + 352);
                    }
                    else
                    {
                      v6 = *(unsigned int *)(a1 + 208);
                      if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
                      {
                        sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), v6 + 1, 8u, v14, v13);
                        v6 = *(unsigned int *)(a1 + 208);
                      }
                      *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v6) = v17;
                      ++*(_DWORD *)(a1 + 208);
                    }
                  }
                }
              }
              goto LABEL_15;
            }
            v38 = 1;
            v23 = 0;
            while ( v15 != -4096 )
            {
              if ( !v23 && v15 == -8192 )
                v23 = (_QWORD *)v6;
              v14 = (v12 - 1) & (v38 + (_DWORD)v14);
              v13 = (unsigned int)(v38 + 1);
              v6 = *(_QWORD *)(a1 + 896) + 16LL * (unsigned int)v14;
              v15 = *(_QWORD *)v6;
              if ( v18 == *(_QWORD *)v6 )
                goto LABEL_6;
              ++v38;
            }
            if ( !v23 )
              v23 = (_QWORD *)v6;
            v24 = *(_DWORD *)(a1 + 904);
            ++*(_QWORD *)(a1 + 888);
            v25 = v24 + 1;
            if ( 4 * v25 < 3 * v12 )
            {
              v14 = v12 >> 3;
              if ( v12 - *(_DWORD *)(a1 + 908) - v25 <= (unsigned int)v14 )
              {
                v39 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
                sub_3512300(a1 + 888, v12);
                v32 = *(_DWORD *)(a1 + 912);
                if ( !v32 )
                {
LABEL_63:
                  ++*(_DWORD *)(a1 + 904);
                  BUG();
                }
                v33 = v32 - 1;
                v34 = 1;
                v31 = 0;
                v14 = *(_QWORD *)(a1 + 896);
                v35 = v33 & v39;
                v25 = *(_DWORD *)(a1 + 904) + 1;
                v23 = (_QWORD *)(v14 + 16LL * (v33 & v39));
                v36 = *v23;
                if ( v18 != *v23 )
                {
                  while ( v36 != -4096 )
                  {
                    if ( v36 == -8192 && !v31 )
                      v31 = v23;
                    v13 = (unsigned int)(v34 + 1);
                    v35 = v33 & (v34 + v35);
                    v23 = (_QWORD *)(v14 + 16LL * v35);
                    v36 = *v23;
                    if ( v18 == *v23 )
                      goto LABEL_34;
                    ++v34;
                  }
                  goto LABEL_42;
                }
              }
              goto LABEL_34;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 888);
          }
          sub_3512300(a1 + 888, 2 * v12);
          v26 = *(_DWORD *)(a1 + 912);
          if ( !v26 )
            goto LABEL_63;
          v27 = v26 - 1;
          v14 = *(_QWORD *)(a1 + 896);
          v28 = v27 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v25 = *(_DWORD *)(a1 + 904) + 1;
          v23 = (_QWORD *)(v14 + 16LL * v28);
          v29 = *v23;
          if ( v18 != *v23 )
          {
            v30 = 1;
            v31 = 0;
            while ( v29 != -4096 )
            {
              if ( !v31 && v29 == -8192 )
                v31 = v23;
              v13 = (unsigned int)(v30 + 1);
              v28 = v27 & (v30 + v28);
              v23 = (_QWORD *)(v14 + 16LL * v28);
              v29 = *v23;
              if ( v18 == *v23 )
                goto LABEL_34;
              ++v30;
            }
LABEL_42:
            if ( v31 )
              v23 = v31;
          }
LABEL_34:
          *(_DWORD *)(a1 + 904) = v25;
          if ( *v23 != -4096 )
            --*(_DWORD *)(a1 + 908);
          *v23 = v18;
          LOBYTE(v6) = 0;
          v16 = 0;
          v23[1] = 0;
          goto LABEL_7;
        }
      }
      else
      {
        v10 = *(_QWORD **)(a5 + 32);
        v11 = (__int64)&v10[*(unsigned int *)(a5 + 40)];
        v6 = (__int64)sub_3510810(v10, v11, v42);
        if ( v11 != v6 )
          goto LABEL_4;
      }
LABEL_15:
      ++v5;
    }
    while ( v9 != v5 );
  }
  return v6;
}
