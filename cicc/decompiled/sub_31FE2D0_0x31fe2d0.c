// Function: sub_31FE2D0
// Address: 0x31fe2d0
//
__int64 __fastcall sub_31FE2D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 *v8; // rbx
  void *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r15d
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // eax
  char v19; // al
  __int64 result; // rax
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // rdi
  int v32; // eax
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  int v38; // [rsp+18h] [rbp-48h]
  unsigned __int64 v39[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a2;
  v6 = *(unsigned int *)(a1 + 8);
  v33 = 0x2E8BA2E8BA2E8BA3LL * ((a3 - a2) >> 3);
  if ( v33 + v6 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v37 = a1 + 16;
    v7 = (__int64 *)sub_C8D7D0(a1, a1 + 16, v33 + v6, 0x58u, v39, a1 + 16);
    sub_31FE0C0((__int64 **)a1, (__int64)v7, v27, v28, v29, v30);
    v31 = *(__int64 **)a1;
    v32 = v39[0];
    if ( v37 != *(_QWORD *)a1 )
    {
      v38 = v39[0];
      _libc_free((unsigned __int64)v31);
      v32 = v38;
    }
    *(_DWORD *)(a1 + 12) = v32;
    v6 = *(unsigned int *)(a1 + 8);
    *(_QWORD *)a1 = v7;
  }
  else
  {
    v7 = *(__int64 **)a1;
  }
  v8 = &v7[11 * v6];
  if ( a2 != a3 )
  {
    do
    {
      while ( 1 )
      {
        if ( v8 )
        {
          v16 = *(_QWORD *)v5;
          v8[1] = 0;
          *((_DWORD *)v8 + 8) = 0;
          *v8 = v16;
          v8[2] = 0;
          *((_DWORD *)v8 + 6) = 0;
          *((_DWORD *)v8 + 7) = 0;
          sub_C7D6A0(0, 0, 4);
          v17 = *(unsigned int *)(v5 + 32);
          *((_DWORD *)v8 + 8) = v17;
          if ( (_DWORD)v17 )
          {
            v9 = (void *)sub_C7D670(12 * v17, 4);
            v8[2] = (__int64)v9;
            *((_DWORD *)v8 + 6) = *(_DWORD *)(v5 + 24);
            *((_DWORD *)v8 + 7) = *(_DWORD *)(v5 + 28);
            memcpy(v9, *(const void **)(v5 + 16), 12LL * *((unsigned int *)v8 + 8));
          }
          else
          {
            v8[2] = 0;
            *((_DWORD *)v8 + 6) = 0;
            *((_DWORD *)v8 + 7) = 0;
          }
          *((_DWORD *)v8 + 12) = 0;
          v8[5] = (__int64)(v8 + 7);
          *((_DWORD *)v8 + 13) = 0;
          v14 = *(_DWORD *)(v5 + 48);
          if ( v14 && v8 + 5 != (__int64 *)(v5 + 40) )
          {
            sub_31FDC20((__int64)(v8 + 5), v14, v10, v11, v12, v13);
            v22 = *(_QWORD *)(v5 + 40);
            v23 = v8[5];
            v24 = v22 + 40LL * *(unsigned int *)(v5 + 48);
            if ( v22 != v24 )
            {
              do
              {
                while ( 1 )
                {
                  if ( v23 )
                  {
                    v25 = *(_QWORD *)v22;
                    *(_DWORD *)(v23 + 16) = 0;
                    *(_DWORD *)(v23 + 20) = 1;
                    *(_QWORD *)v23 = v25;
                    *(_QWORD *)(v23 + 8) = v23 + 24;
                    v26 = *(unsigned int *)(v22 + 16);
                    if ( (_DWORD)v26 )
                      break;
                  }
                  v22 += 40;
                  v23 += 40;
                  if ( v24 == v22 )
                    goto LABEL_24;
                }
                v34 = v24;
                v35 = v22;
                v36 = v23;
                sub_31F3F30(v23 + 8, v22 + 8, v22, v26, v24, v21);
                v24 = v34;
                v22 = v35 + 40;
                v23 = v36 + 40;
              }
              while ( v34 != v35 + 40 );
            }
LABEL_24:
            *((_DWORD *)v8 + 12) = v14;
          }
          v15 = *(_BYTE *)(v5 + 56);
          *((_BYTE *)v8 + 80) = 0;
          *((_BYTE *)v8 + 56) = v15;
          if ( *(_BYTE *)(v5 + 80) )
            break;
        }
        v5 += 88;
        v8 += 11;
        if ( a3 == v5 )
          goto LABEL_15;
      }
      v18 = *(_DWORD *)(v5 + 72);
      *((_DWORD *)v8 + 18) = v18;
      if ( v18 > 0x40 )
        sub_C43780((__int64)(v8 + 8), (const void **)(v5 + 64));
      else
        v8[8] = *(_QWORD *)(v5 + 64);
      v19 = *(_BYTE *)(v5 + 76);
      v5 += 88;
      *((_BYTE *)v8 + 80) = 1;
      v8 += 11;
      *((_BYTE *)v8 - 12) = v19;
    }
    while ( a3 != v5 );
LABEL_15:
    v6 = *(unsigned int *)(a1 + 8);
  }
  result = v33 + v6;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
