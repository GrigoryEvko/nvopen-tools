// Function: sub_D38A70
// Address: 0xd38a70
//
__int64 *__fastcall sub_D38A70(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r14
  __int64 v8; // r13
  unsigned int v9; // edx
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  __int64 v12; // r8
  unsigned __int64 v13; // r10
  unsigned __int64 v14; // rbx
  void *v15; // rdi
  __int64 v16; // rax
  size_t v17; // rdx
  _QWORD *v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // edx
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  int v31; // [rsp+8h] [rbp-58h]
  unsigned int v32; // [rsp+10h] [rbp-50h]
  int v33; // [rsp+10h] [rbp-50h]
  unsigned int v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  unsigned __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_QWORD *)a1;
  v8 = *(unsigned int *)(a1 + 8);
  if ( 3 * v8 )
  {
    v9 = *((_DWORD *)a2 + 2);
    do
    {
      v10 = v9;
      v11 = *a2;
      v12 = v7;
      v13 = v9 + 1LL;
      if ( v13 > *((unsigned int *)a2 + 3) )
      {
        v19 = a2 + 2;
        if ( v11 > v7 || v11 + 48LL * v9 <= v7 )
        {
          v26 = sub_C8D7D0((__int64)a2, (__int64)(a2 + 2), v13, 0x30u, v38, a6);
          sub_D38990(a2, v26, v27, v28, v29, v30);
          v11 = v26;
          if ( v19 == (_QWORD *)*a2 )
          {
            v10 = *((unsigned int *)a2 + 2);
            *((_DWORD *)a2 + 3) = v38[0];
            v12 = v7;
          }
          else
          {
            v33 = v38[0];
            _libc_free(*a2, v26);
            v11 = v26;
            v12 = v7;
            v10 = *((unsigned int *)a2 + 2);
            *((_DWORD *)a2 + 3) = v33;
          }
          *a2 = v26;
          v9 = v10;
        }
        else
        {
          v36 = v7 - v11;
          v20 = sub_C8D7D0((__int64)a2, (__int64)(a2 + 2), v13, 0x30u, v38, a6);
          sub_D38990(a2, v20, v21, v22, v23, v24);
          v25 = v38[0];
          v11 = v20;
          if ( v19 == (_QWORD *)*a2 )
          {
            *a2 = v20;
            *((_DWORD *)a2 + 3) = v25;
          }
          else
          {
            v31 = v38[0];
            _libc_free(*a2, v20);
            v11 = v20;
            *a2 = v20;
            *((_DWORD *)a2 + 3) = v31;
          }
          v10 = *((unsigned int *)a2 + 2);
          v12 = v11 + v36;
          v9 = *((_DWORD *)a2 + 2);
        }
      }
      v14 = v11 + 48 * v10;
      if ( v14 )
      {
        v15 = (void *)(v14 + 32);
        *(_QWORD *)v14 = *(_QWORD *)v12;
        v16 = *(_QWORD *)(v12 + 8);
        *(_QWORD *)(v14 + 16) = v14 + 32;
        *(_QWORD *)(v14 + 8) = v16;
        *(_QWORD *)(v14 + 24) = 0x200000000LL;
        a6 = *(unsigned int *)(v12 + 24);
        if ( (_DWORD)a6 && v14 + 16 != v12 + 16 )
        {
          v17 = 4LL * (unsigned int)a6;
          if ( (unsigned int)a6 <= 2
            || (v34 = *(_DWORD *)(v12 + 24),
                v37 = v12,
                sub_C8D5F0(v14 + 16, (const void *)(v14 + 32), (unsigned int)a6, 4u, v12, a6),
                v12 = v37,
                v15 = *(void **)(v14 + 16),
                a6 = v34,
                (v17 = 4LL * *(unsigned int *)(v37 + 24)) != 0) )
          {
            v32 = a6;
            v35 = v12;
            memcpy(v15, *(const void **)(v12 + 16), v17);
            a6 = v32;
            v12 = v35;
            *(_DWORD *)(v14 + 24) = v32;
          }
          else
          {
            *(_DWORD *)(v14 + 24) = v34;
          }
        }
        *(_DWORD *)(v14 + 40) = *(_DWORD *)(v12 + 40);
        *(_BYTE *)(v14 + 44) = *(_BYTE *)(v12 + 44);
        v9 = *((_DWORD *)a2 + 2);
      }
      ++v9;
      v7 += 48LL;
      *((_DWORD *)a2 + 2) = v9;
      --v8;
    }
    while ( v8 );
  }
  return a2;
}
