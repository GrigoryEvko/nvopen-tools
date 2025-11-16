// Function: sub_13A6D30
// Address: 0x13a6d30
//
__int64 __fastcall sub_13A6D30(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 result; // rax
  __int64 **v4; // rbx
  __int64 **v5; // r15
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r8
  __int64 *v11; // r13
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 **v19; // [rsp+18h] [rbp-38h]

  result = (__int64)&a2[a3];
  v19 = (__int64 **)result;
  if ( (__int64 **)result != a2 )
  {
    v4 = a2;
    v5 = a2;
    v6 = 0;
    do
    {
      v8 = (*v5)[1];
      v9 = sub_1456040(**v5);
      if ( *(_BYTE *)(v9 + 8) == 11 )
      {
        v7 = sub_1456040(v8);
        if ( *(_BYTE *)(v7 + 8) == 11 )
        {
          if ( *(_DWORD *)(v9 + 8) >> 8 > v6 )
          {
            v16 = v9;
            v6 = *(_DWORD *)(v9 + 8) >> 8;
          }
          if ( v6 < *(_DWORD *)(v7 + 8) >> 8 )
          {
            v16 = v7;
            v6 = *(_DWORD *)(v7 + 8) >> 8;
          }
        }
      }
      else
      {
        sub_1456040(v8);
      }
      ++v5;
    }
    while ( v19 != v5 );
    do
    {
      while ( 1 )
      {
        v11 = *v4;
        v12 = **v4;
        v13 = (*v4)[1];
        v17 = sub_1456040(v12);
        if ( *(_BYTE *)(v17 + 8) == 11 )
          break;
        result = sub_1456040(v13);
        if ( v19 == ++v4 )
          return result;
      }
      result = sub_1456040(v13);
      v10 = result;
      if ( *(_BYTE *)(result + 8) == 11 )
      {
        if ( v6 > *(_DWORD *)(v17 + 8) >> 8 )
        {
          v18 = result;
          v14 = sub_147B0D0(*(_QWORD *)(a1 + 8), v12, v16, 0);
          v10 = v18;
          *v11 = v14;
        }
        result = *(_DWORD *)(v10 + 8) >> 8;
        if ( v6 > (unsigned int)result )
        {
          result = sub_147B0D0(*(_QWORD *)(a1 + 8), v13, v16, 0);
          v11[1] = result;
        }
      }
      ++v4;
    }
    while ( v19 != v4 );
  }
  return result;
}
