// Function: sub_35B5B70
// Address: 0x35b5b70
//
__int64 __fastcall sub_35B5B70(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v10; // r15
  unsigned __int64 v11; // rcx
  __int64 v12; // r13
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r8
  _QWORD *v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // [rsp-40h] [rbp-40h]

  result = a2 & 0x7FFFFFFF;
  v7 = result;
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 224) + 32LL) + 4 * result) )
  {
    v8 = 8 * result;
    v10 = *(_QWORD *)(a1 + 232);
    v11 = *(unsigned int *)(v10 + 160);
    if ( (unsigned int)result < (unsigned int)v11 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 8 * result);
      if ( v12 )
        goto LABEL_4;
    }
    v13 = result + 1;
    if ( (unsigned int)v11 < v13 && v13 != v11 )
    {
      if ( v13 >= v11 )
      {
        v17 = *(_QWORD *)(v10 + 168);
        v18 = v13 - v11;
        if ( v13 > (unsigned __int64)*(unsigned int *)(v10 + 164) )
        {
          v21 = v13 - v11;
          sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v13, 8u, v18, a6);
          v11 = *(unsigned int *)(v10 + 160);
          v18 = v21;
        }
        v14 = *(_QWORD *)(v10 + 152);
        v19 = (_QWORD *)(v14 + 8 * v11);
        v20 = &v19[v18];
        if ( v19 != v20 )
        {
          do
            *v19++ = v17;
          while ( v20 != v19 );
          LODWORD(v11) = *(_DWORD *)(v10 + 160);
          v14 = *(_QWORD *)(v10 + 152);
        }
        *(_DWORD *)(v10 + 160) = v18 + v11;
        goto LABEL_7;
      }
      *(_DWORD *)(v10 + 160) = v13;
    }
    v14 = *(_QWORD *)(v10 + 152);
LABEL_7:
    v15 = (__int64 *)(v14 + v8);
    v16 = sub_2E10F30(a2);
    *v15 = v16;
    v12 = v16;
    sub_2E11E80((_QWORD *)v10, v16);
LABEL_4:
    sub_2E21040(*(_QWORD **)(a1 + 240), v12, v7, v11, a5);
    return sub_35B4EE0(a1 + 200, v12);
  }
  return result;
}
