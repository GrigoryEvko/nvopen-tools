// Function: sub_1842FA0
// Address: 0x1842fa0
//
__int64 __fastcall sub_1842FA0(_QWORD *a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v16; // rax
  unsigned __int64 *v17; // rax

  v9 = (_QWORD *)a1[14];
  if ( v9 )
  {
    a5 = a1 + 13;
    v10 = a1 + 13;
    do
    {
      while ( 1 )
      {
        v11 = v9[2];
        v12 = v9[3];
        if ( v9[4] >= a2 )
          break;
        v9 = (_QWORD *)v9[3];
        if ( !v12 )
          goto LABEL_6;
      }
      v10 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v11 );
LABEL_6:
    if ( a5 != v10 && v10[4] <= a2 )
      return 0;
  }
  v13 = (_QWORD *)a1[8];
  if ( v13 )
  {
    v14 = a1 + 7;
    do
    {
      if ( v13[4] < a2
        || v13[4] == a2
        && (*((_DWORD *)v13 + 10) < (unsigned int)a3
         || *((_DWORD *)v13 + 10) == (_DWORD)a3 && *((_BYTE *)v13 + 44) < BYTE4(a3)) )
      {
        v13 = (_QWORD *)v13[3];
      }
      else
      {
        v14 = v13;
        v13 = (_QWORD *)v13[2];
      }
    }
    while ( v13 );
    if ( a1 + 7 != v14
      && v14[4] <= a2
      && (v14[4] != a2
       || (unsigned int)a3 >= *((_DWORD *)v14 + 10)
       && ((_DWORD)a3 != *((_DWORD *)v14 + 10) || BYTE4(a3) >= *((_BYTE *)v14 + 44))) )
    {
      return 0;
    }
  }
  v16 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v16 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, (int)a5, a6);
    v16 = *(unsigned int *)(a4 + 8);
  }
  v17 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v16);
  *v17 = a2;
  v17[1] = a3;
  ++*(_DWORD *)(a4 + 8);
  return 1;
}
