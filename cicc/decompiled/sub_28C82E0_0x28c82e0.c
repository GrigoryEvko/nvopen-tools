// Function: sub_28C82E0
// Address: 0x28c82e0
//
void __fastcall sub_28C82E0(char *a1, char *a2, __int64 a3)
{
  char *v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // r8
  char *v9; // rdi
  char *v10; // r15
  int v11; // edx
  __int64 v12; // r9
  int v13; // edx
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r11
  unsigned int v17; // r11d
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r10
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // eax
  int v25; // eax
  int v26; // r10d
  int v27; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 != a2 && a2 != a1 + 16 )
  {
    v6 = a1 + 32;
    do
    {
      v7 = *((_QWORD *)v6 - 1);
      v8 = *((_QWORD *)a1 + 1);
      v9 = v6 - 16;
      v10 = v6;
      v11 = *(_DWORD *)(a3 + 2376);
      v12 = *(_QWORD *)(a3 + 2360);
      if ( v11 )
      {
        v13 = v11 - 1;
        v14 = v13 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v7 == *v15 )
        {
LABEL_6:
          v17 = *((_DWORD *)v15 + 2);
        }
        else
        {
          v25 = 1;
          while ( v16 != -4096 )
          {
            v26 = v25 + 1;
            v14 = v13 & (v25 + v14);
            v15 = (__int64 *)(v12 + 16LL * v14);
            v16 = *v15;
            if ( v7 == *v15 )
              goto LABEL_6;
            v25 = v26;
          }
          v17 = 0;
        }
        v18 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v19 = (__int64 *)(v12 + 16LL * v18);
        v20 = *v19;
        if ( v8 == *v19 )
        {
LABEL_8:
          if ( *((_DWORD *)v19 + 2) > v17 )
          {
            v21 = *((_QWORD *)v6 - 2);
            v22 = (v9 - a1) >> 4;
            if ( v9 - a1 > 0 )
            {
              do
              {
                v23 = *((_QWORD *)v9 - 2);
                v9 -= 16;
                *((_QWORD *)v9 + 2) = v23;
                *((_QWORD *)v9 + 3) = *((_QWORD *)v9 + 1);
                --v22;
              }
              while ( v22 );
            }
            *(_QWORD *)a1 = v21;
            *((_QWORD *)a1 + 1) = v7;
            goto LABEL_12;
          }
        }
        else
        {
          v24 = 1;
          while ( v20 != -4096 )
          {
            v18 = v13 & (v24 + v18);
            v27 = v24 + 1;
            v19 = (__int64 *)(v12 + 16LL * v18);
            v20 = *v19;
            if ( v8 == *v19 )
              goto LABEL_8;
            v24 = v27;
          }
        }
      }
      sub_28C8130(v9, a3);
LABEL_12:
      v6 += 16;
    }
    while ( a2 != v10 );
  }
}
