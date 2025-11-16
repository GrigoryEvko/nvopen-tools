// Function: sub_2F61E10
// Address: 0x2f61e10
//
void __fastcall sub_2F61E10(__int64 a1, int a2, __int64 **a3, __int64 *a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v10; // esi
  int *v11; // r13
  int v12; // edi
  __int64 *v13; // rbx
  __int64 *v14; // rdx
  __int64 **v15; // rcx
  __int64 *v16; // r14
  int v17; // r12d
  __int64 v18; // r15
  unsigned int v19; // edx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  signed __int64 v24; // rdx
  __int64 v25; // rdi
  bool v26; // dl
  __int64 v27; // rsi
  __int64 v28; // rax
  bool v29; // dl
  __int64 v30; // rax
  _BYTE *v31; // r15
  _BYTE *v32; // rax
  _BYTE *v33; // rbx
  _BYTE *v34; // r12
  __int64 *v35; // r15
  bool v36; // zf
  int v37; // ecx
  __int64 **v38; // [rsp+8h] [rbp-58h]
  __int64 **v39; // [rsp+8h] [rbp-58h]
  __int64 *v40; // [rsp+10h] [rbp-50h]
  __int64 *v41; // [rsp+10h] [rbp-50h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  bool v44; // [rsp+28h] [rbp-38h]
  int v45; // [rsp+28h] [rbp-38h]
  bool v46; // [rsp+2Fh] [rbp-31h]

  v6 = *(unsigned int *)(a1 + 472);
  v7 = *(_QWORD *)(a1 + 456);
  if ( (_DWORD)v6 )
  {
    v10 = (v6 - 1) & (37 * a2);
    v11 = (int *)(v7 + 32LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
    {
LABEL_3:
      if ( v11 != (int *)(v7 + 32 * v6) )
      {
        v13 = *a3;
        v43 = 0;
        v46 = 0;
        v14 = *a3;
        if ( *((_QWORD *)v11 + 1) != *((_QWORD *)v11 + 2) )
        {
          v15 = a3;
          v16 = (__int64 *)*((_QWORD *)v11 + 1);
          v17 = a2;
          while ( 1 )
          {
            if ( v13 == &v14[3 * *((unsigned int *)v15 + 2)] )
              return;
            v18 = *v16;
            v19 = *(_DWORD *)((*v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v16 >> 1) & 3;
            if ( v19 < (*(_DWORD *)((v13[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[1] >> 1) & 3) )
              break;
            v13 += 3;
            if ( *((__int64 **)v11 + 2) == v16 )
              return;
LABEL_9:
            v14 = *v15;
          }
          if ( v19 < (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v13 >> 1) & 3) )
            goto LABEL_7;
          v20 = v16[1];
          v21 = *(_QWORD *)(v20 + 32);
          v22 = v21 + 40;
          if ( *(_WORD *)(v20 + 68) == 14 )
            goto LABEL_34;
          v23 = 40LL * (*(_DWORD *)(v20 + 40) & 0xFFFFFF);
          v22 = v21 + v23;
          v21 += 80;
          v24 = 0xCCCCCCCCCCCCCCCDLL * ((v23 - 80) >> 3);
          v25 = v24 >> 2;
          if ( v24 >> 2 > 0 )
          {
            while ( 1 )
            {
              if ( !*(_BYTE *)v21 && v17 == *(_DWORD *)(v21 + 8) )
              {
                v26 = v21 != v22;
                goto LABEL_38;
              }
              if ( !*(_BYTE *)(v21 + 40) && v17 == *(_DWORD *)(v21 + 48) )
              {
                v26 = v22 != v21 + 40;
                goto LABEL_38;
              }
              if ( !*(_BYTE *)(v21 + 80) && v17 == *(_DWORD *)(v21 + 88) )
              {
                v26 = v22 != v21 + 80;
                goto LABEL_38;
              }
              if ( !*(_BYTE *)(v21 + 120) && v17 == *(_DWORD *)(v21 + 128) )
                break;
              v21 += 160;
              if ( !--v25 )
              {
                v24 = 0xCCCCCCCCCCCCCCCDLL * ((v22 - v21) >> 3);
                goto LABEL_26;
              }
            }
            v36 = v22 == v21 + 120;
            v27 = *v16;
            v26 = !v36;
            if ( v43 != v18 )
              goto LABEL_39;
            goto LABEL_51;
          }
LABEL_26:
          if ( v24 != 2 )
          {
            if ( v24 != 3 )
            {
              if ( v24 != 1 )
              {
                v26 = 0;
                goto LABEL_38;
              }
LABEL_34:
              v26 = 0;
              if ( !*(_BYTE *)v21 && v17 == *(_DWORD *)(v21 + 8) )
LABEL_36:
                v26 = v22 != v21;
LABEL_38:
              v27 = *v16;
              if ( v43 != v18 )
              {
LABEL_39:
                v38 = v15;
                v44 = v26;
                v40 = a4;
                v28 = sub_2E09D00(a4, v27);
                a4 = v40;
                v29 = v44;
                v15 = v38;
                if ( v28 != *v40 + 24LL * *((unsigned int *)v40 + 2) )
                {
                  v43 = v18;
                  v46 = *(_DWORD *)(*(_QWORD *)(a5 + 128) + ((unsigned __int64)**(unsigned int **)(v28 + 16) << 6)) > 1u;
                  v29 = v46 && v44;
                }
LABEL_41:
                if ( v29 )
                {
                  v30 = v16[1];
                  v31 = *(_BYTE **)(v30 + 32);
                  if ( *(_WORD *)(v30 + 68) == 14 )
                  {
                    v32 = v31 + 40;
                  }
                  else
                  {
                    v32 = &v31[40 * (*(_DWORD *)(v30 + 40) & 0xFFFFFF)];
                    v31 += 80;
                  }
                  if ( v31 != v32 )
                  {
                    v41 = v13;
                    v33 = v32;
                    v45 = v17;
                    v34 = v31;
                    v35 = a4;
                    v39 = v15;
                    do
                    {
                      while ( *v34 )
                      {
                        v34 += 40;
                        if ( v33 == v34 )
                          goto LABEL_49;
                      }
                      sub_2EAB0C0((__int64)v34, 0);
                      *(_DWORD *)v34 &= 0xFFF000FF;
                      v34 += 40;
                    }
                    while ( v33 != v34 );
LABEL_49:
                    v13 = v41;
                    v17 = v45;
                    a4 = v35;
                    v15 = v39;
                  }
LABEL_8:
                  if ( *((__int64 **)v11 + 2) == v16 )
                    return;
                  goto LABEL_9;
                }
LABEL_7:
                v16 += 2;
                goto LABEL_8;
              }
LABEL_51:
              v29 = v46 && v26;
              goto LABEL_41;
            }
            if ( !*(_BYTE *)v21 && v17 == *(_DWORD *)(v21 + 8) )
              goto LABEL_36;
            v21 += 40;
          }
          if ( !*(_BYTE *)v21 && v17 == *(_DWORD *)(v21 + 8) )
            goto LABEL_36;
          v21 += 40;
          goto LABEL_34;
        }
      }
    }
    else
    {
      v37 = 1;
      while ( v12 != -1 )
      {
        v10 = (v6 - 1) & (v37 + v10);
        v11 = (int *)(v7 + 32LL * v10);
        v12 = *v11;
        if ( a2 == *v11 )
          goto LABEL_3;
        ++v37;
      }
    }
  }
}
