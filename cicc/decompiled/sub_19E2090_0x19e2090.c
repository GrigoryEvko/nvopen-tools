// Function: sub_19E2090
// Address: 0x19e2090
//
void __fastcall sub_19E2090(char *a1, char *a2, __int64 a3)
{
  char *v6; // rbx
  int v7; // eax
  char *v8; // rdi
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // r10
  char *v12; // r9
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r11
  unsigned int v16; // r15d
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r11
  __int64 v21; // rdi
  __int64 v22; // r8
  char *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // edx
  int v27; // edx
  int v28; // r8d
  int v29; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 != a2 && a2 != a1 + 16 )
  {
    v6 = a1 + 32;
    do
    {
      v7 = *(_DWORD *)(a3 + 2384);
      v8 = v6 - 16;
      if ( v7 )
      {
        v9 = *((_QWORD *)v6 - 1);
        v10 = v7 - 1;
        v11 = *(_QWORD *)(a3 + 2368);
        v12 = v6;
        v13 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v14 = (__int64 *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( v9 == *v14 )
        {
LABEL_6:
          v16 = *((_DWORD *)v14 + 2);
          v17 = *((_QWORD *)a1 + 1);
        }
        else
        {
          v27 = 1;
          while ( v15 != -8 )
          {
            v28 = v27 + 1;
            v13 = v10 & (v27 + v13);
            v14 = (__int64 *)(v11 + 16LL * v13);
            v15 = *v14;
            if ( v9 == *v14 )
              goto LABEL_6;
            v27 = v28;
          }
          v17 = *((_QWORD *)a1 + 1);
          v16 = 0;
        }
        v18 = v10 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v19 = (__int64 *)(v11 + 16LL * v18);
        v20 = *v19;
        if ( *v19 == v17 )
        {
LABEL_8:
          if ( *((_DWORD *)v19 + 2) > v16 )
          {
            v21 = v8 - a1;
            v22 = *((_QWORD *)v6 - 2);
            v23 = v6;
            v24 = v21 >> 4;
            if ( v21 > 0 )
            {
              do
              {
                v25 = *((_QWORD *)v23 - 4);
                v23 -= 16;
                *(_QWORD *)v23 = v25;
                *((_QWORD *)v23 + 1) = *((_QWORD *)v23 - 1);
                --v24;
              }
              while ( v24 );
            }
            *(_QWORD *)a1 = v22;
            *((_QWORD *)a1 + 1) = v9;
            goto LABEL_12;
          }
        }
        else
        {
          v26 = 1;
          while ( v20 != -8 )
          {
            v18 = v10 & (v26 + v18);
            v29 = v26 + 1;
            v19 = (__int64 *)(v11 + 16LL * v18);
            v20 = *v19;
            if ( *v19 == v17 )
              goto LABEL_8;
            v26 = v29;
          }
        }
      }
      sub_19E1F60(v8, a3);
      v12 = v6;
LABEL_12:
      v6 += 16;
    }
    while ( a2 != v12 );
  }
}
