// Function: sub_27A2930
// Address: 0x27a2930
//
void __fastcall sub_27A2930(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r11
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 *v11; // r15
  __int64 v12; // rcx
  int *v13; // rsi
  int *v14; // r12
  int *v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r8
  int v19; // r11d
  int v20; // r10d
  unsigned int v21; // eax
  __int64 v22; // r14
  unsigned int v23; // eax
  int v24; // r8d
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  int v28; // edi
  __int64 v29; // r8
  int *v30; // rcx
  int *v31; // rax
  char v32; // al
  __int64 v33; // rdx
  int *v34; // r9
  unsigned int v35; // eax
  int v36; // eax
  __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 16);
  if ( v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        return;
    }
    v6 = v4;
    while ( 1 )
    {
      v7 = *(_QWORD *)(v5 + 40);
      v8 = *(_QWORD *)(a3 + 8);
      v9 = *(unsigned int *)(a3 + 24);
      if ( (_DWORD)v9 )
      {
        v10 = (v9 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v11 = (__int64 *)(v8 + 88LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
        {
LABEL_6:
          if ( v11 != (__int64 *)(v8 + 88 * v9) )
          {
            v13 = (int *)v11[1];
            v42 = *((unsigned int *)v11 + 4);
            v14 = v13;
            v15 = &v13[8 * v42];
            if ( v15 != v13 )
            {
              v39 = v7;
              while ( *((_QWORD *)v14 + 2) )
              {
                v14 += 8;
LABEL_10:
                if ( v15 == v14 )
                  goto LABEL_19;
              }
              v16 = *(unsigned int *)(a4 + 24);
              v17 = *(_QWORD *)(a4 + 8);
              if ( (_DWORD)v16 )
              {
                v18 = *((_QWORD *)v14 + 1);
                v19 = 1;
                v20 = v16 - 1;
                v21 = (v16 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)((0xBF58476D1CE4E5B9LL * v18) >> 31) ^ (484763065 * (_DWORD)v18)
                        | ((unsigned __int64)(unsigned int)(37 * *v14) << 32))) >> 31)
                     ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v18) >> 31) ^ (484763065 * v18))));
                while ( 1 )
                {
                  v22 = v17 + 48LL * v21;
                  if ( *(_DWORD *)v22 == *v14 && *(_QWORD *)(v22 + 8) == v18 )
                    break;
                  if ( *(_DWORD *)v22 == -1 )
                  {
                    if ( *(_QWORD *)(v22 + 8) == -1 )
                      goto LABEL_28;
                    v35 = v19 + v21;
                    ++v19;
                    v21 = v20 & v35;
                  }
                  else
                  {
                    v23 = v19 + v21;
                    ++v19;
                    v21 = v20 & v23;
                  }
                }
                if ( v22 != 48 * v16 + v17 && *(_DWORD *)(v22 + 24) )
                {
                  sub_B196A0(
                    *(_QWORD *)(a1 + 216),
                    v39,
                    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v22 + 16) + 8LL * *(unsigned int *)(v22 + 24) - 8) + 40LL));
                  if ( v32 )
                  {
                    *((_QWORD *)v14 + 2) = a2;
                    v33 = *(_QWORD *)(*(_QWORD *)(v22 + 16) + 8LL * (unsigned int)(*(_DWORD *)(v22 + 24))-- - 8);
                    *((_QWORD *)v14 + 3) = v33;
                  }
                  v13 = (int *)v11[1];
                  v42 = *((unsigned int *)v11 + 4);
                }
              }
LABEL_28:
              v25 = (char *)&v13[8 * v42] - (char *)v14;
              v26 = v25 >> 7;
              v27 = v25 >> 5;
              if ( v26 > 0 )
              {
                v28 = *v14;
                v29 = *((_QWORD *)v14 + 1);
                v30 = v14;
                v31 = &v14[32 * v26];
                while ( *((_QWORD *)v30 + 1) == v29 )
                {
                  v34 = v30 + 8;
                  if ( v28 != v30[8]
                    || v29 != *((_QWORD *)v30 + 5)
                    || (v34 = v30 + 16, v28 != v30[16])
                    || v29 != *((_QWORD *)v30 + 9)
                    || (v34 = v30 + 24, v28 != v30[24])
                    || v29 != *((_QWORD *)v30 + 13) )
                  {
                    v30 = v34;
                    goto LABEL_31;
                  }
                  v30 += 32;
                  if ( v31 == v30 )
                  {
                    v27 = ((char *)&v13[8 * v42] - (char *)v30) >> 5;
                    goto LABEL_40;
                  }
                  if ( v28 != *v30 )
                    goto LABEL_31;
                }
                goto LABEL_31;
              }
              v30 = v14;
LABEL_40:
              switch ( v27 )
              {
                case 2LL:
                  v36 = *v14;
                  break;
                case 3LL:
                  v36 = *v14;
                  if ( *v30 != *v14 || *((_QWORD *)v30 + 1) != *((_QWORD *)v14 + 1) )
                    goto LABEL_31;
                  v30 += 8;
                  break;
                case 1LL:
                  v36 = *v14;
LABEL_55:
                  if ( *v30 == v36 && *((_QWORD *)v30 + 1) == *((_QWORD *)v14 + 1) )
                    v30 = &v13[8 * v42];
                  goto LABEL_31;
                default:
                  v30 = &v13[8 * v42];
LABEL_31:
                  v14 = v30;
                  goto LABEL_10;
              }
              if ( *v30 != v36 || *((_QWORD *)v30 + 1) != *((_QWORD *)v14 + 1) )
                goto LABEL_31;
              v30 += 8;
              goto LABEL_55;
            }
          }
        }
        else
        {
          v24 = 1;
          while ( v12 != -4096 )
          {
            v10 = (v9 - 1) & (v24 + v10);
            v11 = (__int64 *)(v8 + 88LL * v10);
            v12 = *v11;
            if ( v7 == *v11 )
              goto LABEL_6;
            ++v24;
          }
        }
      }
LABEL_19:
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return;
      while ( 1 )
      {
        v5 = *(_QWORD *)(v6 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
  }
}
