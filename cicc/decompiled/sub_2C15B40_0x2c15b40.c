// Function: sub_2C15B40
// Address: 0x2c15b40
//
__int64 __fastcall sub_2C15B40(__int64 a1)
{
  __int64 *v1; // rax
  int v2; // edx
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r12
  char v7; // r14
  unsigned int *v8; // r15
  __int64 v9; // r9
  unsigned int v10; // r14d
  unsigned int v11; // eax
  unsigned int v12; // edx
  __int64 v13; // rdi
  unsigned int v14; // ecx
  unsigned int v15; // edx
  unsigned int v16; // esi
  int *v17; // rax
  int v18; // r10d
  __int64 v19; // rbx
  __int64 v20; // rdi
  __int64 v21; // r8
  __int64 v22; // r8
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  int v31; // eax
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 *v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h] BYREF
  __int64 v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = *(__int64 **)(a1 + 48);
  v2 = *(_DWORD *)(a1 + 56);
  v36 = v1 + 1;
  v35 = (unsigned int)(v2 - 1);
  if ( *(_BYTE *)(a1 + 104) )
  {
    v3 = v1[v2 - 1];
    v35 = (unsigned int)(v2 - 2);
  }
  else
  {
    v3 = 0;
  }
  v32 = *v1;
  v4 = sub_22077B0(0x70u);
  v6 = v4;
  if ( v4 )
  {
    v7 = *(_BYTE *)(a1 + 105);
    v37 = v32;
    v8 = *(unsigned int **)(a1 + 96);
    v38[0] = 0;
    sub_2AAF310(v4, 5, &v37, 1, v38, v5);
    sub_9C6650(v38);
    *(_BYTE *)(v6 + 105) = v7;
    v10 = 0;
    *(_QWORD *)(v6 + 96) = v8;
    *(_BYTE *)(v6 + 104) = 0;
    *(_QWORD *)(v6 + 40) = &unk_4A249D8;
    v11 = *v8;
    *(_QWORD *)v6 = &unk_4A24998;
    if ( v11 )
    {
      do
      {
        v12 = v8[8];
        v13 = *((_QWORD *)v8 + 2);
        v14 = v10 + v8[10];
        if ( v12 )
        {
          v15 = v12 - 1;
          v16 = v15 & (37 * v14);
          v17 = (int *)(v13 + 16LL * v16);
          v18 = *v17;
          if ( v14 == *v17 )
          {
LABEL_7:
            v19 = *((_QWORD *)v17 + 1);
            if ( v19 )
            {
              if ( *(_BYTE *)(*(_QWORD *)(v19 + 8) + 8LL) != 7 )
              {
                v20 = sub_22077B0(0x38u);
                if ( v20 )
                  sub_2BF0340(v20, 0, v19, v6, v21, v9);
              }
            }
          }
          else
          {
            v31 = 1;
            while ( v18 != 0x7FFFFFFF )
            {
              v9 = (unsigned int)(v31 + 1);
              v16 = v15 & (v31 + v16);
              v17 = (int *)(v13 + 16LL * v16);
              v18 = *v17;
              if ( v14 == *v17 )
                goto LABEL_7;
              v31 = v9;
            }
          }
        }
        ++v10;
      }
      while ( *v8 > v10 );
    }
    v22 = (__int64)&v36[v35];
    if ( (__int64 *)v22 != v36 )
    {
      v23 = v36;
      do
      {
        v24 = *(unsigned int *)(v6 + 56);
        v25 = *v23;
        if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 60) )
        {
          v34 = v22;
          sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v24 + 1, 8u, v22, v9);
          v24 = *(unsigned int *)(v6 + 56);
          v22 = v34;
        }
        *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v24) = v25;
        ++*(_DWORD *)(v6 + 56);
        v26 = *(unsigned int *)(v25 + 24);
        if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v25 + 28) )
        {
          v33 = v22;
          sub_C8D5F0(v25 + 16, (const void *)(v25 + 32), v26 + 1, 8u, v22, v9);
          v26 = *(unsigned int *)(v25 + 24);
          v22 = v33;
        }
        ++v23;
        *(_QWORD *)(*(_QWORD *)(v25 + 16) + 8 * v26) = v6 + 40;
        ++*(_DWORD *)(v25 + 24);
      }
      while ( (__int64 *)v22 != v23 );
    }
    if ( v3 )
    {
      v27 = *(unsigned int *)(v6 + 56);
      v28 = *(unsigned int *)(v6 + 60);
      *(_BYTE *)(v6 + 104) = 1;
      if ( v27 + 1 > v28 )
      {
        sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v27 + 1, 8u, v22, v9);
        v27 = *(unsigned int *)(v6 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v27) = v3;
      ++*(_DWORD *)(v6 + 56);
      v29 = *(unsigned int *)(v3 + 24);
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v3 + 28) )
      {
        sub_C8D5F0(v3 + 16, (const void *)(v3 + 32), v29 + 1, 8u, v22, v9);
        v29 = *(unsigned int *)(v3 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * v29) = v6 + 40;
      ++*(_DWORD *)(v3 + 24);
    }
  }
  return v6;
}
