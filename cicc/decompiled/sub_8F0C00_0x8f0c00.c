// Function: sub_8F0C00
// Address: 0x8f0c00
//
void *__fastcall sub_8F0C00(__int64 a1, int a2)
{
  __int64 v2; // rcx
  __int64 v3; // r12
  void *v4; // r13
  __int64 v5; // rax
  int v6; // r14d
  int v7; // r12d
  char *v8; // rbx
  __int64 v9; // rax
  void *v10; // rdi
  __int64 v11; // rdx
  _QWORD *v12; // r13
  __int64 v13; // rax
  void *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  void *result; // rax
  __int64 v19; // rcx
  size_t v20; // rdx
  int v21; // r12d
  __int64 v22; // rbx
  __int64 v23; // rax
  void *dest; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  char *v27; // [rsp+20h] [rbp-40h]
  int v28; // [rsp+28h] [rbp-38h]
  int v29; // [rsp+2Ch] [rbp-34h]

  v2 = qword_4F690E0;
  dest = (void *)(a1 + 8);
  v3 = *(_QWORD *)qword_4F690E0;
  v4 = (void *)(qword_4F690E0 + 8);
  v26 = qword_4F690E0;
  *(_DWORD *)(qword_4F690E0 + 2088) = 0;
  v5 = *(unsigned int *)(a1 + 2088);
  qword_4F690E0 = v3;
  *(_DWORD *)(v2 + 2088) = v5;
  memcpy(v4, (const void *)(a1 + 8), 4 * v5);
  if ( (a2 & 3) != 0 )
    sub_8EEC10(v26, dword_3C237F0[a2 & 3]);
  v29 = a2 >> 2;
  if ( a2 >> 2 )
  {
    v28 = 0;
    v27 = (char *)&unk_4F62EA0;
    do
    {
      if ( (v29 & 1) != 0 )
      {
        v6 = dword_4F62E90;
        if ( !dword_4F62E90 )
        {
          dword_4F62EA8 = 625;
          v6 = v29 & 1;
          dword_4F636C8 = 1;
          dword_4F62E90 = 1;
        }
        v7 = 11;
        if ( v28 <= 11 )
          v7 = v28;
        if ( v7 >= v6 )
        {
          v8 = (char *)&unk_4F62EA0 + 2096 * v6 - 2096;
          do
          {
            ++v6;
            v9 = sub_8F0A50((__int64)v8, (__int64)v8);
            v10 = v8 + 2104;
            v8 += 2096;
            v11 = *(unsigned int *)(v9 + 2088);
            v12 = (_QWORD *)v9;
            *((_DWORD *)v8 + 522) = v11;
            memcpy(v10, (const void *)(v9 + 8), 4 * v11);
            v13 = qword_4F690E0;
            dword_4F62E90 = v6;
            qword_4F690E0 = (__int64)v12;
            *v12 = v13;
          }
          while ( v7 >= v6 );
        }
        if ( v6 <= v28 )
        {
          v19 = 2096LL * (v6 - 1);
          v20 = 4LL * *(unsigned int *)((char *)&unk_4F62EA0 + v19 + 2088);
          dword_4F62E88 = *(_DWORD *)((char *)&unk_4F62EA0 + v19 + 2088);
          memcpy(&unk_4F62668, (char *)&unk_4F62EA0 + v19 + 8, v20);
          v21 = v28 - (v6 - 1);
          if ( v28 != v6 - 1 )
          {
            do
            {
              v22 = sub_8F0A50((__int64)&unk_4F62660, (__int64)&unk_4F62660);
              dword_4F62E88 = *(_DWORD *)(v22 + 2088);
              memcpy(&unk_4F62668, (const void *)(v22 + 8), 4LL * (unsigned int)dword_4F62E88);
              v23 = qword_4F690E0;
              qword_4F690E0 = v22;
              *(_QWORD *)v22 = v23;
              --v21;
            }
            while ( v21 );
          }
          v14 = &unk_4F62660;
        }
        else
        {
          v14 = v27;
        }
        v15 = sub_8F0A50(v26, (__int64)v14);
        v16 = qword_4F690E0;
        qword_4F690E0 = v26;
        v26 = v15;
        *(_QWORD *)qword_4F690E0 = v16;
      }
      ++v28;
      v27 += 2096;
      v29 >>= 1;
    }
    while ( v29 );
    v3 = qword_4F690E0;
    v4 = (void *)(v26 + 8);
  }
  v17 = *(unsigned int *)(v26 + 2088);
  *(_DWORD *)(a1 + 2088) = v17;
  result = memcpy(dest, v4, 4 * v17);
  *(_QWORD *)v26 = v3;
  qword_4F690E0 = v26;
  return result;
}
