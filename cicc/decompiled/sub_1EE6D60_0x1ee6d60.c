// Function: sub_1EE6D60
// Address: 0x1ee6d60
//
void __fastcall sub_1EE6D60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  int *v8; // r13
  unsigned __int64 v9; // r14
  int i; // eax
  int v11; // edx
  _BYTE *v12; // rsi
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  _BYTE *v16; // rcx
  __int64 v17; // rax
  int *v18; // r15
  const void *v19; // rsi
  int v20; // eax
  int *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  int *v25; // r10
  int *v26; // r14
  int *v27; // r15
  int v28; // r13d
  int v31; // [rsp+10h] [rbp-40h]
  bool v32; // [rsp+17h] [rbp-39h]

  v5 = a1;
  if ( *(_DWORD *)(a1 + 88) )
  {
    v8 = *(int **)(a1 + 80);
    v32 = a5 != 0;
    v9 = a4 & 0xFFFFFFFFFFFFFFF8LL | 6;
    for ( i = sub_1EE4FF0(a2, a3, 1, *v8, v9, 0xFFFFFFFF, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_1EE56A0);
          ;
          i = sub_1EE4FF0(a2, a3, 1, *v8, v9, 0xFFFFFFFF, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_1EE56A0) )
    {
      v11 = v8[1];
      if ( *v8 < 0 && v32 && (i & ~v11) == 0 )
      {
        v31 = i;
        sub_1E1A650(a5, *v8, 1);
        v11 = v8[1];
        i = v31;
      }
      v12 = v8 + 2;
      v13 = v11 & i;
      if ( v13 )
      {
        v8[1] = v13;
        v8 += 2;
        if ( v12 == (_BYTE *)(*(_QWORD *)(a1 + 80) + 8LL * *(unsigned int *)(a1 + 88)) )
          goto LABEL_11;
      }
      else
      {
        v14 = *(_QWORD *)(a1 + 80);
        v15 = *(unsigned int *)(a1 + 88);
        v16 = (_BYTE *)(v14 + 8 * v15);
        if ( v16 != v12 )
        {
          memmove(v8, v12, v16 - v12);
          LODWORD(v15) = *(_DWORD *)(a1 + 88);
          v14 = *(_QWORD *)(a1 + 80);
        }
        v17 = (unsigned int)(v15 - 1);
        *(_DWORD *)(a1 + 88) = v17;
        if ( v8 == (int *)(v14 + 8 * v17) )
        {
LABEL_11:
          v5 = a1;
          break;
        }
      }
    }
  }
  v18 = *(int **)v5;
  if ( *(_DWORD *)(v5 + 8) )
  {
    do
    {
      while ( 1 )
      {
        v19 = v18 + 2;
        v20 = v18[1]
            & sub_1EE4FF0(
                a2,
                a3,
                1,
                *v18,
                a4 & 0xFFFFFFFFFFFFFFF8LL,
                0xFFFFFFFF,
                (unsigned __int8 (__fastcall *)(__int64, __int64))sub_1EE56A0);
        if ( !v20 )
          break;
        v18[1] = v20;
        v18 += 2;
        if ( v19 == (const void *)(*(_QWORD *)v5 + 8LL * *(unsigned int *)(v5 + 8)) )
          goto LABEL_19;
      }
      v21 = *(int **)v5;
      v22 = *(unsigned int *)(v5 + 8);
      v23 = *(_QWORD *)v5 + 8 * v22;
      if ( (const void *)v23 != v19 )
      {
        memmove(v18, v19, v23 - (_QWORD)v19);
        LODWORD(v22) = *(_DWORD *)(v5 + 8);
        v21 = *(int **)v5;
      }
      v24 = (unsigned int)(v22 - 1);
      *(_DWORD *)(v5 + 8) = v24;
    }
    while ( v18 != &v21[2 * v24] );
  }
LABEL_19:
  if ( a5 )
  {
    v25 = *(int **)(v5 + 160);
    v26 = &v25[2 * *(unsigned int *)(v5 + 168)];
    if ( v26 != v25 )
    {
      v27 = *(int **)(v5 + 160);
      do
      {
        while ( 1 )
        {
          v28 = *v27;
          if ( *v27 < 0
            && !(unsigned int)sub_1EE4FF0(
                                a2,
                                a3,
                                1,
                                v28,
                                a4 & 0xFFFFFFFFFFFFFFF8LL | 6,
                                0xFFFFFFFF,
                                (unsigned __int8 (__fastcall *)(__int64, __int64))sub_1EE56A0) )
          {
            break;
          }
          v27 += 2;
          if ( v26 == v27 )
            return;
        }
        v27 += 2;
        sub_1E1A650(a5, v28, 1);
      }
      while ( v26 != v27 );
    }
  }
}
