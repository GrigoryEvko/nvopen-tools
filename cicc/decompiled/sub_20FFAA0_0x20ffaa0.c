// Function: sub_20FFAA0
// Address: 0x20ffaa0
//
__int64 __fastcall sub_20FFAA0(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v6; // r8d
  int v7; // r9d
  unsigned int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rax
  __int64 v14; // r13
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned int v17; // ebx
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rdx

  v8 = sub_1E6B9A0(
         a1[3],
         *(_QWORD *)(*(_QWORD *)(a1[3] + 24LL) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
         (unsigned __int8 *)byte_3F871B3,
         0,
         a5,
         a6);
  v9 = a1[5];
  if ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 312);
    v11 = *(_DWORD *)(v10 + 4LL * (a2 & 0x7FFFFFFF));
    if ( !v11 )
      v11 = a2;
    *(_DWORD *)(v10 + 4LL * (v8 & 0x7FFFFFFF)) = v11;
  }
  v12 = a1[1];
  if ( v12 && *(float *)(v12 + 116) == INFINITY )
  {
    v14 = a1[4];
    v15 = v8 & 0x7FFFFFFF;
    v16 = *(unsigned int *)(v14 + 408);
    if ( (v8 & 0x7FFFFFFF) < (unsigned int)v16 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(v14 + 400) + 8LL * v15);
      if ( v19 )
        goto LABEL_12;
    }
    v17 = v15 + 1;
    if ( (unsigned int)v16 < v15 + 1 )
    {
      v20 = v17;
      if ( v17 >= v16 )
      {
        if ( v17 > v16 )
        {
          if ( v17 > (unsigned __int64)*(unsigned int *)(v14 + 412) )
          {
            sub_16CD150(v14 + 400, (const void *)(v14 + 416), v17, 8, v6, v7);
            v16 = *(unsigned int *)(v14 + 408);
            v20 = v17;
          }
          v18 = *(_QWORD *)(v14 + 400);
          v21 = (_QWORD *)(v18 + 8 * v20);
          v22 = (_QWORD *)(v18 + 8 * v16);
          v23 = *(_QWORD *)(v14 + 416);
          if ( v21 != v22 )
          {
            do
              *v22++ = v23;
            while ( v21 != v22 );
            v18 = *(_QWORD *)(v14 + 400);
          }
          *(_DWORD *)(v14 + 408) = v17;
          goto LABEL_11;
        }
      }
      else
      {
        *(_DWORD *)(v14 + 408) = v17;
      }
    }
    v18 = *(_QWORD *)(v14 + 400);
LABEL_11:
    *(_QWORD *)(v18 + 8LL * (v8 & 0x7FFFFFFF)) = sub_1DBA290(v8);
    v19 = *(_QWORD *)(*(_QWORD *)(v14 + 400) + 8LL * (v8 & 0x7FFFFFFF));
    sub_1DBB110((_QWORD *)v14, v19);
LABEL_12:
    *(_DWORD *)(v19 + 116) = 2139095040;
  }
  return v8;
}
