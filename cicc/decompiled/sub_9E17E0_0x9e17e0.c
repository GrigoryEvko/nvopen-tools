// Function: sub_9E17E0
// Address: 0x9e17e0
//
_QWORD *__fastcall sub_9E17E0(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4, char a5, char a6, char a7)
{
  _QWORD *v7; // r15
  int v9; // r13d
  char v13; // al
  _QWORD *v14; // r14
  unsigned int v15; // r12d
  char v16; // r15
  unsigned int v17; // r10d
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned int v29; // [rsp+1Ch] [rbp-44h]
  _QWORD *v30; // [rsp+20h] [rbp-40h]

  v7 = a1;
  v9 = a4;
  v30 = a1 + 2;
  *a1 = a1 + 2;
  a1[1] = 0;
  if ( !a5 && (a6 || a7) )
  {
    if ( a4 >> 1 )
      sub_C8D5F0(a1, v30, a4 >> 1, 16);
  }
  else
  {
    if ( !a4 )
      return v7;
    sub_C8D5F0(a1, v30, a4, 16);
  }
  if ( v9 )
  {
    v13 = a6;
    v14 = v7;
    v15 = 0;
    v16 = v13;
    do
    {
      v25 = sub_9E1590(a2, *(_QWORD *)(a3 + 8LL * v15));
      v17 = v15 + 1;
      v26 = v25;
      if ( a5 )
      {
        if ( v16 )
        {
          v17 = v15 + 2;
          LODWORD(v18) = 0;
          LOBYTE(v19) = 0;
          LOBYTE(v20) = 0;
        }
        else
        {
          LOBYTE(v20) = 0;
          LODWORD(v18) = 0;
          LOBYTE(v19) = 0;
        }
      }
      else if ( v16 )
      {
        LODWORD(v18) = 0;
        v19 = *(_QWORD *)(a3 + 8LL * v17) & 7LL;
        v20 = (*(_QWORD *)(a3 + 8LL * v17) >> 3) & 1LL;
      }
      else if ( a7 )
      {
        LOBYTE(v19) = 0;
        v18 = *(_QWORD *)(a3 + 8LL * v17) & 0xFFFFFFFLL;
        v20 = (*(_QWORD *)(a3 + 8LL * v17) >> 28) & 1LL;
      }
      else
      {
        LOBYTE(v20) = 0;
        v17 = v15;
        LODWORD(v18) = 0;
        LOBYTE(v19) = 0;
      }
      v21 = (unsigned __int8)v19 | (8 * (unsigned __int8)v20);
      v22 = *((unsigned int *)v14 + 2);
      v23 = (16 * (_DWORD)v18) | (unsigned int)v21;
      if ( v22 + 1 > (unsigned __int64)*((unsigned int *)v14 + 3) )
      {
        v27 = v23;
        v28 = v26;
        v29 = v17;
        sub_C8D5F0(v14, v30, v22 + 1, 16);
        v22 = *((unsigned int *)v14 + 2);
        v23 = v27;
        v26 = v28;
        v17 = v29;
      }
      v24 = (__int64 *)(*v14 + 16 * v22);
      v15 = v17 + 1;
      *v24 = v26;
      v24[1] = v23;
      ++*((_DWORD *)v14 + 2);
    }
    while ( v9 != v17 + 1 );
    return v14;
  }
  return v7;
}
