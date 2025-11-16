// Function: sub_615770
// Address: 0x615770
//
__int64 __fastcall sub_615770(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rsi
  unsigned int v5; // ecx
  int v6; // eax
  __int64 result; // rax
  unsigned int v8; // ebx
  unsigned int v9; // r15d
  __int64 v10; // r12
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r14
  __int64 v15; // r8
  __int64 *v16; // r12
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rax
  unsigned int v20; // ecx
  __int64 *i; // rax
  __int64 *v22; // [rsp+0h] [rbp-50h]
  unsigned int v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]

  v4 = (_QWORD *)(*a1 + 16LL * a2);
  *v4 = a3;
  if ( a3 )
    v4[1] = a4;
  v5 = *((_DWORD *)a1 + 2);
  v6 = *((_DWORD *)a1 + 3) + 1;
  *((_DWORD *)a1 + 3) = v6;
  result = (unsigned int)(2 * v6);
  if ( (unsigned int)result > v5 )
  {
    v23 = v5;
    v8 = v5 + 1;
    v9 = 2 * v5 + 1;
    v10 = 2 * v5 + 2;
    v11 = (_QWORD *)sub_822B10(16 * v10);
    v13 = v23;
    v14 = v11;
    if ( (_DWORD)v10 )
    {
      v12 = (__int64)&v11[2 * v9 + 2];
      do
      {
        if ( v11 )
          *v11 = 0;
        v11 += 2;
      }
      while ( (_QWORD *)v12 != v11 );
    }
    v15 = *a1;
    if ( v8 )
    {
      v13 = 16LL * v23;
      v16 = (__int64 *)*a1;
      v12 = v15 + v13 + 16;
      do
      {
        while ( 1 )
        {
          if ( *v16 )
          {
            v22 = (__int64 *)v12;
            v24 = v15;
            v17 = sub_722DF0();
            v18 = sub_887620(v17);
            v15 = v24;
            v12 = (__int64)v22;
            v19 = v9 & v18;
            v20 = v19;
            for ( i = &v14[2 * v19]; *i; i = &v14[2 * v20] )
              v20 = v9 & (v20 + 1);
            v13 = *v16;
            *i = *v16;
            if ( v13 )
              break;
          }
          v16 += 2;
          if ( (__int64 *)v12 == v16 )
            goto LABEL_17;
        }
        v13 = v16[1];
        v16 += 2;
        i[1] = v13;
      }
      while ( v22 != v16 );
    }
LABEL_17:
    *a1 = (__int64)v14;
    *((_DWORD *)a1 + 2) = v9;
    return sub_822B90(v15, 16LL * v8, v12, v13);
  }
  return result;
}
