// Function: sub_350B2D0
// Address: 0x350b2d0
//
__int64 __fastcall sub_350B2D0(_QWORD *a1, int a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v9; // eax
  __int64 v10; // rdi
  int v11; // r12d
  int v12; // edx
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  unsigned int v15; // edx
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v20; // r12
  unsigned int v21; // eax
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *i; // rbx
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r15
  _QWORD *v28; // rax
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 *v31; // r15
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 *v35; // rdx
  __int64 *v36; // rdi
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // rax
  __int64 *v40; // rsi
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  __int64 v43; // [rsp+10h] [rbp-40h]
  __int64 v44; // [rsp+10h] [rbp-40h]
  __int64 v45; // [rsp+18h] [rbp-38h]
  __int64 v46; // [rsp+18h] [rbp-38h]
  __int64 v47; // [rsp+18h] [rbp-38h]

  v9 = sub_2EC0780(a1[3], a2, byte_3F871B3, 0, a5, a6);
  v10 = a1[5];
  v11 = v9;
  if ( v10 )
  {
    v12 = *(_DWORD *)(*(_QWORD *)(v10 + 80) + 4LL * (a2 & 0x7FFFFFFF));
    if ( !v12 )
      v12 = a2;
    sub_350ABD0(v10, v9, v12);
  }
  v13 = a1[4];
  v14 = *(unsigned int *)(v13 + 160);
  v15 = (v11 & 0x7FFFFFFF) + 1;
  if ( v15 <= (unsigned int)v14 || v15 == v14 )
    goto LABEL_6;
  if ( v15 < v14 )
  {
    *(_DWORD *)(v13 + 160) = v15;
LABEL_6:
    v16 = *(_QWORD *)(v13 + 152);
    goto LABEL_7;
  }
  v33 = *(_QWORD *)(v13 + 168);
  v34 = v15 - v14;
  if ( v15 > (unsigned __int64)*(unsigned int *)(v13 + 164) )
  {
    v41 = v15 - v14;
    v43 = *(_QWORD *)(v13 + 168);
    v46 = a1[4];
    sub_C8D5F0(v13 + 152, (const void *)(v13 + 168), v15, 8u, v33, v34);
    v13 = v46;
    v34 = v41;
    v33 = v43;
    v14 = *(unsigned int *)(v46 + 160);
  }
  v16 = *(_QWORD *)(v13 + 152);
  v35 = (__int64 *)(v16 + 8 * v14);
  v36 = &v35[v34];
  if ( v35 != v36 )
  {
    do
      *v35++ = v33;
    while ( v36 != v35 );
    LODWORD(v14) = *(_DWORD *)(v13 + 160);
    v16 = *(_QWORD *)(v13 + 152);
  }
  *(_DWORD *)(v13 + 160) = v34 + v14;
LABEL_7:
  v17 = sub_2E10F30(v11);
  *(_QWORD *)(v16 + 8LL * (v11 & 0x7FFFFFFF)) = v17;
  v18 = a1[1];
  if ( v18 && *(float *)(v18 + 116) == INFINITY )
    *(_DWORD *)(v17 + 116) = 2139095040;
  if ( a3 )
  {
    v20 = a1[4];
    v21 = a2 & 0x7FFFFFFF;
    v22 = *(unsigned int *)(v20 + 160);
    if ( (a2 & 0x7FFFFFFFu) < (unsigned int)v22 )
    {
      v23 = *(_QWORD *)(*(_QWORD *)(v20 + 152) + 8LL * v21);
      if ( v23 )
      {
LABEL_17:
        for ( i = *(_QWORD **)(v23 + 104); i; i = (_QWORD *)i[13] )
        {
          v25 = *(_QWORD *)(v20 + 56);
          v26 = i[14];
          v27 = i[15];
          *(_QWORD *)(v20 + 136) += 128LL;
          v28 = (_QWORD *)((v25 + 15) & 0xFFFFFFFFFFFFFFF0LL);
          if ( *(_QWORD *)(v20 + 64) >= (unsigned __int64)(v28 + 16) && v25 )
          {
            *(_QWORD *)(v20 + 56) = v28 + 16;
            if ( !v28 )
            {
              MEMORY[0x68] = *(_QWORD *)(v17 + 104);
              BUG();
            }
          }
          else
          {
            v42 = v26;
            v28 = (_QWORD *)sub_9D1E70(v20 + 56, 128, 128, 4);
            v26 = v42;
          }
          v28[13] = 0;
          *v28 = v28 + 2;
          v28[1] = 0x200000000LL;
          v28[8] = v28 + 10;
          v28[9] = 0x200000000LL;
          v28[12] = 0;
          v28[14] = v26;
          v28[15] = v27;
          v28[13] = *(_QWORD *)(v17 + 104);
          *(_QWORD *)(v17 + 104) = v28;
        }
        return v17;
      }
    }
    v29 = v21 + 1;
    if ( (unsigned int)v22 < v29 && v29 != v22 )
    {
      if ( v29 >= v22 )
      {
        v37 = *(_QWORD *)(v20 + 168);
        v38 = v29 - v22;
        if ( v29 > (unsigned __int64)*(unsigned int *)(v20 + 164) )
        {
          v44 = v29 - v22;
          v47 = *(_QWORD *)(v20 + 168);
          sub_C8D5F0(v20 + 152, (const void *)(v20 + 168), v29, 8u, v37, v38);
          v38 = v44;
          v37 = v47;
          v22 = *(unsigned int *)(v20 + 160);
        }
        v30 = *(_QWORD *)(v20 + 152);
        v39 = (__int64 *)(v30 + 8 * v22);
        v40 = &v39[v38];
        if ( v39 != v40 )
        {
          do
            *v39++ = v37;
          while ( v40 != v39 );
          LODWORD(v22) = *(_DWORD *)(v20 + 160);
          v30 = *(_QWORD *)(v20 + 152);
        }
        *(_DWORD *)(v20 + 160) = v38 + v22;
        goto LABEL_26;
      }
      *(_DWORD *)(v20 + 160) = v29;
    }
    v30 = *(_QWORD *)(v20 + 152);
LABEL_26:
    v31 = (__int64 *)(v30 + 8LL * (a2 & 0x7FFFFFFF));
    v32 = sub_2E10F30(a2);
    *v31 = v32;
    v45 = v32;
    sub_2E11E80((_QWORD *)v20, v32);
    v20 = a1[4];
    v23 = v45;
    goto LABEL_17;
  }
  return v17;
}
