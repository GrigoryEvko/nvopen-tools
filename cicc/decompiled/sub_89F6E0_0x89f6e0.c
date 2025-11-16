// Function: sub_89F6E0
// Address: 0x89f6e0
//
void __fastcall sub_89F6E0(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r13
  __int64 v6; // rsi
  __int64 v7; // r15
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 *v13; // r9
  __int64 v14; // r8
  _QWORD *v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-38h]

  v5 = a1[2];
  if ( v5 < a2 )
  {
    v6 = a1[1];
    v7 = *a1;
    if ( a2 > v6 )
    {
      v10 = (_QWORD *)sub_823970(8 * a2);
      v14 = (__int64)v10;
      if ( v5 > 0 )
      {
        v11 = (__int64 *)v7;
        v15 = &v10[v5];
        do
        {
          if ( v10 )
          {
            v12 = *v11;
            *v10 = *v11;
          }
          ++v10;
          ++v11;
        }
        while ( v10 != v15 );
      }
      v16 = v14;
      sub_823A00(v7, 8 * v6, (__int64)v11, v12, v14, v13);
      a1[1] = a2;
      *a1 = v16;
      v7 = v16;
    }
    v9 = (_QWORD *)(v7 + 8 * v5);
    do
    {
      if ( v9 )
        *v9 = *a3;
      ++v9;
      ++a1[2];
    }
    while ( v9 != (_QWORD *)(v7 + 8 * a2) );
  }
  else if ( v5 > a2 )
  {
    a1[2] = a2;
  }
}
