// Function: sub_1AFCA80
// Address: 0x1afca80
//
void __fastcall sub_1AFCA80(__int64 *src, char *a2, __int64 *a3)
{
  unsigned __int64 v5; // rax
  char *v6; // r15
  __int64 v7; // rbx
  char *v8; // rdi
  unsigned int v9; // r14d
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // r9d
  __int64 v13; // r8
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r10
  __int64 *v17; // rax
  unsigned int v18; // r11d
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // r10
  int v22; // edx
  int v23; // edx
  int v24; // r11d
  int v25; // [rsp-3Ch] [rbp-3Ch]

  if ( src != (__int64 *)a2 )
  {
    _BitScanReverse64(&v5, (a2 - (char *)src) >> 3);
    sub_1AFC5D0(src, a2, 2LL * (int)(63 - (v5 ^ 0x3F)), a3);
    if ( a2 - (char *)src <= 128 )
    {
      sub_1AFC210((char *)src, a2, (__int64)a3);
    }
    else
    {
      v6 = (char *)(src + 16);
      sub_1AFC210((char *)src, (char *)src + 128, (__int64)a3);
      if ( a2 != (char *)(src + 16) )
      {
        do
        {
          v7 = *(_QWORD *)v6;
          v8 = v6;
          v9 = ((unsigned int)*(_QWORD *)v6 >> 9) ^ ((unsigned int)*(_QWORD *)v6 >> 4);
          while ( 1 )
          {
            v10 = *((_QWORD *)v8 - 1);
            v11 = *(unsigned int *)(*a3 + 48);
            if ( !(_DWORD)v11 )
              goto LABEL_25;
            v12 = v11 - 1;
            v13 = *(_QWORD *)(*a3 + 32);
            v14 = (v11 - 1) & v9;
            v15 = (__int64 *)(v13 + 16LL * v14);
            v16 = *v15;
            if ( v7 != *v15 )
            {
              v23 = 1;
              while ( v16 != -8 )
              {
                v24 = v23 + 1;
                v14 = v12 & (v23 + v14);
                v15 = (__int64 *)(v13 + 16LL * v14);
                v16 = *v15;
                if ( v7 == *v15 )
                  goto LABEL_8;
                v23 = v24;
              }
LABEL_25:
              BUG();
            }
LABEL_8:
            v17 = (__int64 *)(v13 + 16 * v11);
            if ( v17 == v15 )
              BUG();
            v18 = *(_DWORD *)(v15[1] + 16);
            v19 = v12 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v20 = (__int64 *)(v13 + 16LL * v19);
            v21 = *v20;
            if ( v10 != *v20 )
            {
              v22 = 1;
              while ( v21 != -8 )
              {
                v19 = v12 & (v22 + v19);
                v25 = v22 + 1;
                v20 = (__int64 *)(v13 + 16LL * v19);
                v21 = *v20;
                if ( v10 == *v20 )
                  goto LABEL_10;
                v22 = v25;
              }
LABEL_26:
              BUG();
            }
LABEL_10:
            if ( v20 == v17 )
              goto LABEL_26;
            if ( v18 <= *(_DWORD *)(v20[1] + 16) )
              break;
            *(_QWORD *)v8 = v10;
            v8 -= 8;
          }
          v6 += 8;
          *(_QWORD *)v8 = v7;
        }
        while ( a2 != v6 );
      }
    }
  }
}
