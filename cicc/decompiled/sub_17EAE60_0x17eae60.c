// Function: sub_17EAE60
// Address: 0x17eae60
//
void __fastcall sub_17EAE60(__int64 a1)
{
  __int64 v1; // r15
  int v2; // edi
  unsigned int v3; // edx
  __int64 *v4; // rax
  __int64 v5; // r8
  __int64 v6; // rbx
  unsigned __int64 v7; // r13
  int v8; // r9d
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // r14
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  _BYTE *v14; // rdx
  _BYTE *v15; // rdi
  __int64 v16; // rbx
  unsigned __int64 v17; // r13
  unsigned int v18; // r12d
  __int64 v19; // r15
  __int64 *v20; // r14
  __int64 v21; // rsi
  int v22; // eax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r12
  __int64 v25; // rax
  unsigned int v26; // r14d
  int i; // r10d
  int v28; // eax
  __int64 *v29; // r10
  unsigned __int64 v30; // [rsp+8h] [rbp-78h]
  unsigned int v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+28h] [rbp-58h]
  _QWORD *v35; // [rsp+30h] [rbp-50h] BYREF
  __int64 v36; // [rsp+38h] [rbp-48h]
  _BYTE v37[64]; // [rsp+40h] [rbp-40h] BYREF

  v1 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v34 = *(_QWORD *)a1 + 72LL;
  while ( v34 != v1 )
  {
    while ( 1 )
    {
      v6 = v1 - 24;
      if ( !v1 )
        v6 = 0;
      v7 = sub_157EBA0(v6);
      if ( (unsigned int)sub_15F4D60(v7) <= 1 || (unsigned __int8)(*(_BYTE *)(v7 + 16) - 26) > 2u )
        goto LABEL_5;
      v9 = *(unsigned int *)(a1 + 296);
      v10 = *(_QWORD *)(a1 + 280);
      if ( (_DWORD)v9 )
        break;
      v11 = *(_QWORD *)(v10 + 8);
      if ( *(_QWORD *)(v11 + 16) )
        goto LABEL_12;
LABEL_5:
      v1 = *(_QWORD *)(v1 + 8);
      if ( v34 == v1 )
        return;
    }
    v2 = v9 - 1;
    v3 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v4 = (__int64 *)(v10 + 16LL * v3);
    v5 = *v4;
    if ( v6 == *v4 )
    {
      if ( !*(_QWORD *)(v4[1] + 16) )
        goto LABEL_5;
LABEL_25:
      v11 = v4[1];
    }
    else
    {
      v25 = *v4;
      v26 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      for ( i = 1; ; i = v8 )
      {
        if ( v25 == -8 )
        {
          if ( !*(_QWORD *)(*(_QWORD *)(v10 + 16LL * (unsigned int)v9 + 8) + 16LL) )
            goto LABEL_5;
          goto LABEL_31;
        }
        v8 = i + 1;
        v26 = v2 & (i + v26);
        v29 = (__int64 *)(v10 + 16LL * v26);
        v25 = *v29;
        if ( v6 == *v29 )
          break;
      }
      if ( !*(_QWORD *)(v29[1] + 16) )
        goto LABEL_5;
LABEL_31:
      v28 = 1;
      while ( v5 != -8 )
      {
        v8 = v28 + 1;
        v3 = v2 & (v28 + v3);
        v4 = (__int64 *)(v10 + 16LL * v3);
        v5 = *v4;
        if ( v6 == *v4 )
          goto LABEL_25;
        v28 = v8;
      }
      v11 = *(_QWORD *)(v10 + 16 * v9 + 8);
    }
LABEL_12:
    v12 = *(unsigned int *)(v11 + 80);
    v35 = v37;
    v13 = v12;
    v31 = v12;
    v36 = 0x200000000LL;
    if ( v12 > 2 )
    {
      sub_16CD150((__int64)&v35, v37, v12, 8, v5, v8);
      v15 = v35;
      v14 = &v35[v13];
      LODWORD(v36) = v31;
      if ( &v35[v13] == v35 )
        goto LABEL_16;
    }
    else
    {
      LODWORD(v36) = v12;
      v14 = &v37[v13 * 8];
      v15 = v37;
      if ( &v37[v13 * 8] == v37 )
        goto LABEL_15;
    }
    memset(v15, 0, v14 - v15);
LABEL_15:
    if ( !v31 )
    {
      v24 = 0;
      goto LABEL_22;
    }
LABEL_16:
    v30 = v7;
    v16 = 0;
    v17 = 0;
    v18 = v31;
    v32 = v1;
    v19 = v11;
    do
    {
      v20 = *(__int64 **)(*(_QWORD *)(v19 + 72) + 8 * v16);
      v21 = v20[1];
      if ( v21 )
      {
        v22 = sub_137DFF0(*v20, v21);
        v23 = v20[4];
        v35[v22] = v23;
        if ( v17 < v23 )
          v17 = v23;
      }
      ++v16;
    }
    while ( v18 > (unsigned int)v16 );
    v24 = v17;
    v1 = v32;
    v7 = v30;
LABEL_22:
    sub_17E9890(*(__int64 **)(a1 + 8), v7, v35, (unsigned int)v36, v24, v8);
    if ( v35 == (_QWORD *)v37 )
      goto LABEL_5;
    _libc_free((unsigned __int64)v35);
    v1 = *(_QWORD *)(v1 + 8);
  }
}
