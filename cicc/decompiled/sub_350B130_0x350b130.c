// Function: sub_350B130
// Address: 0x350b130
//
__int64 __fastcall sub_350B130(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // r9
  __int64 v9; // rdi
  unsigned int v10; // r13d
  int v11; // edx
  __int64 v12; // rax
  __int64 v14; // r14
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rbx
  unsigned __int64 v23; // r15
  _QWORD *v24; // rax
  _QWORD *v25; // rsi

  v7 = sub_2EC0780(a1[3], a2, byte_3F871B3, 0, a5, a6);
  v9 = a1[5];
  v10 = v7;
  if ( v9 )
  {
    v11 = *(_DWORD *)(*(_QWORD *)(v9 + 80) + 4LL * (a2 & 0x7FFFFFFF));
    if ( !v11 )
      v11 = a2;
    sub_350ABD0(v9, v7, v11);
  }
  v12 = a1[1];
  if ( v12 && *(float *)(v12 + 116) == INFINITY )
  {
    v14 = a1[4];
    v15 = v10 & 0x7FFFFFFF;
    v16 = *(unsigned int *)(v14 + 160);
    if ( (v10 & 0x7FFFFFFF) < (unsigned int)v16 )
    {
      v21 = *(_QWORD *)(*(_QWORD *)(v14 + 152) + 8LL * v15);
      if ( v21 )
        goto LABEL_12;
    }
    v17 = v15 + 1;
    if ( (unsigned int)v16 < v17 && v17 != v16 )
    {
      if ( v17 >= v16 )
      {
        v22 = *(_QWORD *)(v14 + 168);
        v23 = v17 - v16;
        if ( v17 > (unsigned __int64)*(unsigned int *)(v14 + 164) )
        {
          sub_C8D5F0(v14 + 152, (const void *)(v14 + 168), v17, 8u, v17, v8);
          v16 = *(unsigned int *)(v14 + 160);
        }
        v18 = *(_QWORD *)(v14 + 152);
        v24 = (_QWORD *)(v18 + 8 * v16);
        v25 = &v24[v23];
        if ( v24 != v25 )
        {
          do
            *v24++ = v22;
          while ( v25 != v24 );
          LODWORD(v16) = *(_DWORD *)(v14 + 160);
          v18 = *(_QWORD *)(v14 + 152);
        }
        *(_DWORD *)(v14 + 160) = v23 + v16;
        goto LABEL_11;
      }
      *(_DWORD *)(v14 + 160) = v17;
    }
    v18 = *(_QWORD *)(v14 + 152);
LABEL_11:
    v19 = (__int64 *)(v18 + 8LL * (v10 & 0x7FFFFFFF));
    v20 = sub_2E10F30(v10);
    *v19 = v20;
    v21 = v20;
    sub_2E11E80((_QWORD *)v14, v20);
LABEL_12:
    *(_DWORD *)(v21 + 116) = 2139095040;
  }
  return v10;
}
