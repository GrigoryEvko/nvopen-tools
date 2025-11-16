// Function: sub_2F65A90
// Address: 0x2f65a90
//
__int64 __fastcall sub_2F65A90(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r10
  __int64 result; // rax
  __int64 v8; // r15
  unsigned int v9; // ecx
  unsigned __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r10
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdi
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 **v26; // rdi
  __int64 v27; // r8
  unsigned __int64 v28; // r13
  __int64 *v29; // rdx
  __int64 *v30; // rcx
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  int v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  unsigned int v35; // [rsp+20h] [rbp-50h] BYREF
  int v36; // [rsp+24h] [rbp-4Ch] BYREF
  int v37; // [rsp+28h] [rbp-48h] BYREF
  int v38; // [rsp+2Ch] [rbp-44h] BYREF
  int v39; // [rsp+30h] [rbp-40h] BYREF
  int v40; // [rsp+34h] [rbp-3Ch] BYREF
  int v41; // [rsp+38h] [rbp-38h] BYREF
  int v42[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v3 = a1[3];
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  if ( !(unsigned __int8)sub_2F61710(v3, a2, &v35, &v36, &v37, &v38)
    || (unsigned int)(v36 - 1) <= 0x3FFFFFFE
    || v35 - 1 <= 0x3FFFFFFE
    || !(unsigned __int8)sub_2F612C0(v36, a2, a1[2]) )
  {
    return 0;
  }
  v8 = a1[5];
  v9 = v5 & 0x7FFFFFFF;
  v10 = *(unsigned int *)(v8 + 160);
  v34 = *(_QWORD *)(a2 + 24);
  v11 = 8 * (v5 & 0x7FFFFFFF);
  if ( ((unsigned int)v5 & 0x7FFFFFFF) >= (unsigned int)v10
    || (v12 = *(_QWORD *)(*(_QWORD *)(v8 + 152) + 8LL * v9)) == 0 )
  {
    v16 = v9 + 1;
    if ( (unsigned int)v10 < v9 + 1 && v16 != v10 )
    {
      if ( v16 >= v10 )
      {
        v27 = *(_QWORD *)(v8 + 168);
        v28 = v16 - v10;
        if ( v16 > (unsigned __int64)*(unsigned int *)(v8 + 164) )
        {
          v31 = *(_QWORD *)(v8 + 168);
          v33 = v5;
          sub_C8D5F0(v8 + 152, (const void *)(v8 + 168), v16, 8u, v27, v5);
          v10 = *(unsigned int *)(v8 + 160);
          v27 = v31;
          LODWORD(v5) = v33;
        }
        v17 = *(_QWORD *)(v8 + 152);
        v29 = (__int64 *)(v17 + 8 * v10);
        v30 = &v29[v28];
        if ( v29 != v30 )
        {
          do
            *v29++ = v27;
          while ( v30 != v29 );
          v17 = *(_QWORD *)(v8 + 152);
        }
        *(_DWORD *)(v8 + 160) += v28;
        goto LABEL_26;
      }
      *(_DWORD *)(v8 + 160) = v16;
    }
    v17 = *(_QWORD *)(v8 + 152);
LABEL_26:
    v18 = (__int64 *)(v17 + v11);
    v19 = sub_2E10F30(v5);
    *v18 = v19;
    v12 = v19;
    sub_2E11E80((_QWORD *)v8, v19);
    v6 = a1[2];
    v4 = v35;
  }
  if ( (int)v4 < 0 )
    v13 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16 * (v4 & 0x7FFFFFFF) + 8);
  else
    v13 = *(_QWORD *)(*(_QWORD *)(v6 + 304) + 8 * v4);
  if ( !v13 )
    return 0;
  while ( (*(_BYTE *)(v13 + 4) & 8) != 0 )
  {
    v13 = *(_QWORD *)(v13 + 32);
    if ( !v13 )
      return 0;
  }
  v14 = *(_QWORD *)(v13 + 16);
LABEL_14:
  if ( a2 == v14 || ((*(_WORD *)(v14 + 68) - 12) & 0xFFF7) != 0 || v34 != *(_QWORD *)(v14 + 24) )
  {
LABEL_17:
    v15 = v14;
    goto LABEL_19;
  }
  v20 = a1[3];
  v32 = v14;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42[0] = 0;
  result = sub_2F61710(v20, a2, &v39, &v40, &v41, v42);
  if ( (_BYTE)result )
  {
    v21 = v40;
    if ( v40 == v35 )
    {
      v21 = v39;
      v40 = v39;
    }
    if ( (unsigned int)(v21 - 1) <= 0x3FFFFFFE || (unsigned __int8)sub_2F612C0(v40, v32, a1[2]) )
    {
      v14 = *(_QWORD *)(v13 + 16);
      goto LABEL_17;
    }
    v26 = (__int64 **)sub_2DF8570(a1[5], v24, v22, v23, v24, v25);
    if ( !*(_DWORD *)(v12 + 8) || (result = sub_2E09D90(v26, v12, *(__int64 **)v12), !(_BYTE)result) )
    {
      v15 = *(_QWORD *)(v13 + 16);
LABEL_19:
      while ( 1 )
      {
        v13 = *(_QWORD *)(v13 + 32);
        if ( !v13 )
          return 0;
        if ( (*(_BYTE *)(v13 + 4) & 8) == 0 )
        {
          v14 = *(_QWORD *)(v13 + 16);
          if ( v15 != v14 )
            goto LABEL_14;
        }
      }
    }
  }
  return result;
}
