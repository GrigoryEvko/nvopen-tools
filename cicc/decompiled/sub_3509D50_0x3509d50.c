// Function: sub_3509D50
// Address: 0x3509d50
//
__int64 __fastcall sub_3509D50(__int64 a1)
{
  __int64 v2; // rax
  __int64 *v3; // r14
  __int64 result; // rax
  __int64 *v5; // r12
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r9
  int v14; // r10d
  unsigned int v15; // eax
  __int64 v16; // rdx
  int v17; // ecx
  __int64 v18; // r8
  __int64 v19; // rbx
  unsigned __int64 v20; // rcx
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // r13
  unsigned __int64 v26; // rax
  _QWORD *v27; // rdx
  _QWORD *v28; // rdi
  unsigned __int64 v29; // [rsp+0h] [rbp-50h]
  int v30; // [rsp+Ch] [rbp-44h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  _QWORD *v33; // [rsp+18h] [rbp-38h]
  __int64 v34; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(__int64 **)(v2 + 64);
  result = *(unsigned int *)(v2 + 72);
  v5 = &v3[result];
  if ( v5 != v3 )
  {
    while ( 1 )
    {
      v13 = *v3;
      if ( (*(_QWORD *)(*v3 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        break;
LABEL_9:
      if ( v5 == ++v3 )
        goto LABEL_20;
    }
    v14 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 112LL);
    v15 = v14 & 0x7FFFFFFF;
    v16 = v14 & 0x7FFFFFFF;
    v17 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 80LL) + 4 * v16);
    if ( v17 )
    {
      v14 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 80LL) + 4 * v16);
      v15 = v17 & 0x7FFFFFFF;
      v16 = v17 & 0x7FFFFFFF;
    }
    v18 = *(_QWORD *)(a1 + 32);
    v19 = 8 * v16;
    v20 = *(unsigned int *)(v18 + 160);
    if ( (unsigned int)v20 > v15 )
    {
      v6 = *(_QWORD *)(*(_QWORD *)(v18 + 152) + 8 * v16);
      if ( v6 )
      {
LABEL_4:
        v7 = *(_QWORD *)(v13 + 8);
        v8 = (__int64 *)sub_2E09D00((__int64 *)v6, v7);
        result = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
        if ( v8 != (__int64 *)result )
        {
          result = *(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3;
          if ( (unsigned int)result <= (*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7 >> 1) & 3) )
          {
            v11 = v8[2];
            if ( v11 )
            {
              result = *(_QWORD *)(v11 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              v12 = *(_QWORD *)(result + 16);
              if ( v12 )
                result = sub_3509C80(a1, v11, v12, v7 & 0xFFFFFFFFFFFFFFF8LL, v9, v10);
            }
          }
        }
        goto LABEL_9;
      }
    }
    v21 = v15 + 1;
    if ( (unsigned int)v20 < v21 )
    {
      v24 = v21;
      if ( v21 != v20 )
      {
        if ( v21 >= v20 )
        {
          v25 = *(_QWORD *)(v18 + 168);
          v26 = v21 - v20;
          if ( v24 > *(unsigned int *)(v18 + 164) )
          {
            v29 = v26;
            v30 = v14;
            v32 = *v3;
            v34 = *(_QWORD *)(a1 + 32);
            sub_C8D5F0(v18 + 152, (const void *)(v18 + 168), v24, 8u, v18, v13);
            v18 = v34;
            v26 = v29;
            v14 = v30;
            v13 = v32;
            v20 = *(unsigned int *)(v34 + 160);
          }
          v22 = *(_QWORD *)(v18 + 152);
          v27 = (_QWORD *)(v22 + 8 * v20);
          v28 = &v27[v26];
          if ( v27 != v28 )
          {
            do
              *v27++ = v25;
            while ( v28 != v27 );
            LODWORD(v20) = *(_DWORD *)(v18 + 160);
            v22 = *(_QWORD *)(v18 + 152);
          }
          *(_DWORD *)(v18 + 160) = v26 + v20;
          goto LABEL_16;
        }
        *(_DWORD *)(v18 + 160) = v21;
      }
    }
    v22 = *(_QWORD *)(v18 + 152);
LABEL_16:
    v31 = v13;
    v33 = (_QWORD *)v18;
    v23 = sub_2E10F30(v14);
    *(_QWORD *)(v22 + v19) = v23;
    v6 = v23;
    sub_2E11E80(v33, v23);
    v13 = v31;
    goto LABEL_4;
  }
LABEL_20:
  *(_BYTE *)(a1 + 68) = 1;
  return result;
}
