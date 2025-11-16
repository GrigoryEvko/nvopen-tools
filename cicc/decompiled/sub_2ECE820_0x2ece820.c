// Function: sub_2ECE820
// Address: 0x2ece820
//
__int64 __fastcall sub_2ECE820(_QWORD *a1, __int64 a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int *v11; // r8
  __int64 v12; // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned __int16 *v16; // rdi
  unsigned __int16 *v17; // rax
  __int64 v18; // rdx
  unsigned __int16 v19; // cx
  __int64 v20; // r12
  unsigned int v22; // r12d
  unsigned int v23; // eax
  unsigned int v24; // ebx
  int v25; // eax
  unsigned int *v26; // r12
  __int64 v27; // rax
  unsigned int v28; // r8d
  __int64 v29; // r14
  __int64 v30; // rax
  unsigned int *v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  unsigned int v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  unsigned int v35; // [rsp+18h] [rbp-38h]

  v8 = a1[1];
  v9 = *(unsigned int *)(a1[45] + 4LL * a3);
  v10 = *(_QWORD *)(v8 + 32) + 32LL * a3;
  v11 = *(unsigned int **)(v10 + 24);
  LODWORD(v34) = *(_DWORD *)(v10 + 8);
  if ( !v11 || *(_DWORD *)(v10 + 16) )
  {
    LODWORD(v32) = 0;
    v22 = -1;
    v35 = v9 + v34;
    if ( (unsigned int)v9 < v35 )
    {
      do
      {
        v23 = sub_2ECE620((__int64)a1, v9, a4, a5);
        if ( v23 < v22 )
        {
          LODWORD(v32) = v9;
          v22 = v23;
        }
        LODWORD(v9) = v9 + 1;
      }
      while ( v35 != (_DWORD)v9 );
    }
    return (v32 << 32) | v22;
  }
  else
  {
    v12 = *(_QWORD *)(v8 + 192);
    v14 = *(unsigned __int16 *)(a2 + 2);
    v15 = *(_QWORD *)(v12 + 176);
    v16 = (unsigned __int16 *)(v15 + 6 * (v14 + *(unsigned __int16 *)(a2 + 4)));
    v17 = (unsigned __int16 *)(v15 + 6 * v14);
    if ( v17 == v16 )
    {
LABEL_15:
      v24 = -1;
      if ( (_DWORD)v34 )
      {
        v25 = v34;
        LODWORD(v34) = 0;
        v26 = v11;
        v27 = (__int64)&v11[v25 - 1 + 1];
        v28 = a5;
        v29 = a2;
        v31 = (unsigned int *)v27;
        do
        {
          v33 = v28;
          v30 = sub_2ECE820(a1, v29, *v26, a4);
          v28 = v33;
          if ( (unsigned int)v30 < v24 )
          {
            LODWORD(v34) = HIDWORD(v30);
            v24 = v30;
          }
          ++v26;
        }
        while ( v31 != v26 );
      }
      return (v34 << 32) | v24;
    }
    else
    {
      v18 = a1[55] + 16LL * a3;
      while ( 1 )
      {
        v19 = *v17;
        v20 = *(_QWORD *)v18;
        if ( *(_DWORD *)(v18 + 8) > 0x40u )
          v20 = *(_QWORD *)(*(_QWORD *)v18 + 8 * ((unsigned __int64)v19 >> 6));
        if ( (v20 & (1LL << v19)) != 0 )
          return (v9 << 32) | (unsigned int)sub_2ECE620((__int64)a1, v9, a4, a5);
        v17 += 3;
        if ( v16 == v17 )
          goto LABEL_15;
      }
    }
  }
}
