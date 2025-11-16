// Function: sub_1398490
// Address: 0x1398490
//
__int64 __fastcall sub_1398490(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  int v4; // r15d
  unsigned int v6; // r12d
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp-40h] [rbp-40h]
  __int64 v15; // [rsp-40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  result = (*(_QWORD *)(a1 + 16) - v2) >> 5;
  if ( (_DWORD)result )
  {
    v4 = (*(_QWORD *)(a1 + 16) - v2) >> 5;
    v6 = 0;
    while ( 1 )
    {
      result = 32LL * v6;
      if ( *(_QWORD *)(v2 + result + 24) == a2 )
      {
        --*(_DWORD *)(a2 + 32);
        v7 = *(_QWORD *)(a1 + 16);
        v8 = *(_QWORD *)(a1 + 8) + result;
        v9 = *(_QWORD *)(v8 + 16);
        v10 = v8;
        v11 = *(_QWORD *)(v7 - 16);
        if ( v9 != v11 )
        {
          if ( v9 != -8 && v9 != 0 && v9 != -16 )
          {
            v14 = *(_QWORD *)(a1 + 16);
            sub_1649B30(v10);
            v7 = v14;
            v11 = *(_QWORD *)(v14 - 16);
          }
          *(_QWORD *)(v10 + 16) = v11;
          if ( v11 != -8 && v11 != 0 && v11 != -16 )
          {
            v15 = v7;
            sub_1649AC0(v10, *(_QWORD *)(v7 - 32) & 0xFFFFFFFFFFFFFFF8LL);
            v7 = v15;
          }
        }
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(v7 - 8);
        v12 = *(_QWORD *)(a1 + 16);
        v13 = v12 - 32;
        *(_QWORD *)(a1 + 16) = v12 - 32;
        result = *(_QWORD *)(v12 - 16);
        if ( result != 0 && result != -8 && result != -16 )
          result = sub_1649B30(v13);
        if ( --v4 == v6 )
          return result;
      }
      else if ( v4 == ++v6 )
      {
        return result;
      }
      v2 = *(_QWORD *)(a1 + 8);
    }
  }
  return result;
}
