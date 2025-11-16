// Function: sub_2DF8570
// Address: 0x2df8570
//
__int64 __fastcall sub_2DF8570(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // r13
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // r15
  _QWORD *v18; // rcx
  _QWORD *v19; // rsi

  v6 = a2 & 0x7FFFFFFF;
  v8 = 8LL * (a2 & 0x7FFFFFFF);
  v9 = *(unsigned int *)(a1 + 160);
  if ( (unsigned int)v9 <= (a2 & 0x7FFFFFFF) || (v10 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL * v6)) == 0 )
  {
    v12 = v6 + 1;
    if ( (unsigned int)v9 < v12 && v12 != v9 )
    {
      if ( v12 >= v9 )
      {
        v16 = *(_QWORD *)(a1 + 168);
        v17 = v12 - v9;
        if ( v12 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
        {
          sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v12, 8u, v12, a6);
          v9 = *(unsigned int *)(a1 + 160);
        }
        v13 = *(_QWORD *)(a1 + 152);
        v18 = (_QWORD *)(v13 + 8 * v9);
        v19 = &v18[v17];
        if ( v18 != v19 )
        {
          do
            *v18++ = v16;
          while ( v19 != v18 );
          LODWORD(v9) = *(_DWORD *)(a1 + 160);
          v13 = *(_QWORD *)(a1 + 152);
        }
        *(_DWORD *)(a1 + 160) = v17 + v9;
        goto LABEL_6;
      }
      *(_DWORD *)(a1 + 160) = v12;
    }
    v13 = *(_QWORD *)(a1 + 152);
LABEL_6:
    v14 = (__int64 *)(v13 + v8);
    v15 = sub_2E10F30(a2);
    *v14 = v15;
    v10 = v15;
    sub_2E11E80(a1, v15);
  }
  return v10;
}
