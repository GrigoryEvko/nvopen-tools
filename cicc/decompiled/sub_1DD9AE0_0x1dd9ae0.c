// Function: sub_1DD9AE0
// Address: 0x1dd9ae0
//
signed __int64 __fastcall sub_1DD9AE0(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 *v4; // r10
  __int64 v5; // r15
  __int64 *v7; // rbx
  __int64 v8; // r9
  __int64 v9; // rdi
  __int64 *v10; // rax
  unsigned int v11; // ecx
  unsigned int v12; // edx
  unsigned int v13; // esi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 *v17; // rsi
  unsigned __int64 v18; // r12
  int v19; // r11d
  __int64 *v20; // rax
  unsigned __int64 v21; // r13
  unsigned int v22; // ecx
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 *v33; // [rsp-40h] [rbp-40h]

  result = a2 - (_QWORD)a1;
  if ( (__int64)(a2 - (_QWORD)a1) <= 256 )
    return result;
  v4 = (__int64 *)a2;
  v5 = a3;
  if ( !a3 )
  {
    v21 = a2;
    goto LABEL_23;
  }
  v7 = a1 + 2;
  v33 = a1 + 4;
  while ( 2 )
  {
    v8 = a1[2];
    --v5;
    v9 = *a1;
    v10 = &a1[2 * (result >> 5)];
    v11 = *(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v8 >> 1) & 3;
    v12 = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v10 >> 1) & 3;
    v13 = *(_DWORD *)((*(v4 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(v4 - 2) >> 1) & 3;
    if ( v11 >= v12 )
    {
      if ( v11 < v13 )
        goto LABEL_7;
      if ( v12 < v13 )
      {
LABEL_17:
        *a1 = *(v4 - 2);
        v27 = *(v4 - 1);
        *(v4 - 2) = v9;
        v28 = a1[1];
        a1[1] = v27;
        *(v4 - 1) = v28;
        v9 = a1[2];
        goto LABEL_8;
      }
LABEL_21:
      *a1 = *v10;
      v29 = v10[1];
      *v10 = v9;
      v30 = a1[1];
      a1[1] = v29;
      v10[1] = v30;
      v9 = a1[2];
      goto LABEL_8;
    }
    if ( v12 < v13 )
      goto LABEL_21;
    if ( v11 < v13 )
      goto LABEL_17;
LABEL_7:
    v14 = a1[1];
    v15 = a1[3];
    *a1 = v8;
    a1[2] = v9;
    a1[1] = v15;
    a1[3] = v14;
LABEL_8:
    v16 = *a1;
    v17 = v33;
    v18 = (unsigned __int64)v7;
    v19 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    v20 = v4;
    while ( 1 )
    {
      v21 = v18;
      v22 = v19 | (v16 >> 1) & 3;
      if ( (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v9 >> 1) & 3) < v22 )
        goto LABEL_14;
      v23 = *(v20 - 2);
      v20 -= 2;
      if ( (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v23 >> 1) & 3) > v22 )
      {
        do
        {
          v24 = *(v20 - 2);
          v20 -= 2;
        }
        while ( v22 < (*(_DWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v24 >> 1) & 3) );
      }
      if ( v18 >= (unsigned __int64)v20 )
        break;
      *(v17 - 2) = *v20;
      v25 = v20[1];
      *v20 = v9;
      v26 = *(v17 - 1);
      *(v17 - 1) = v25;
      v20[1] = v26;
      v16 = *a1;
      v19 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_14:
      v9 = *v17;
      v18 += 16LL;
      v17 += 2;
    }
    sub_1DD9AE0(v18, v4, v5);
    result = v18 - (_QWORD)a1;
    if ( (__int64)(v18 - (_QWORD)a1) > 256 )
    {
      if ( v5 )
      {
        v4 = (__int64 *)v18;
        continue;
      }
LABEL_23:
      sub_1DD99D0(a1, (char *)v21, v21);
      do
      {
        v21 -= 16LL;
        v31 = *(_QWORD *)v21;
        v32 = *(_QWORD *)(v21 + 8);
        *(_QWORD *)v21 = *a1;
        *(_QWORD *)(v21 + 8) = a1[1];
        result = (signed __int64)sub_1DD5430((__int64)a1, 0, (__int64)(v21 - (_QWORD)a1) >> 4, v31, v32);
      }
      while ( (__int64)(v21 - (_QWORD)a1) > 16 );
    }
    return result;
  }
}
