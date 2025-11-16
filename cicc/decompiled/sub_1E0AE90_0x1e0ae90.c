// Function: sub_1E0AE90
// Address: 0x1e0ae90
//
__int64 __fastcall sub_1E0AE90(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  void *v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r12
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  _WORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != result )
  {
    v4 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0xEu )
    {
      result = sub_16E7EE0(a2, "Constant Pool:\n", 0xFu);
    }
    else
    {
      result = 0x746E6174736E6F43LL;
      qmemcpy(v4, "Constant Pool:\n", 15);
      *(_QWORD *)(a2 + 24) += 15LL;
    }
    v5 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 4;
    if ( (_DWORD)v5 )
    {
      v6 = 0;
      do
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(a2 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v9) <= 4 )
          {
            v10 = sub_16E7EE0(a2, "  cp#", 5u);
          }
          else
          {
            *(_DWORD *)v9 = 1885544480;
            v10 = a2;
            *(_BYTE *)(v9 + 4) = 35;
            *(_QWORD *)(a2 + 24) += 5LL;
          }
          v11 = sub_16E7A90(v10, v6);
          v12 = *(_WORD **)(v11 + 24);
          if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 1u )
          {
            sub_16E7EE0(v11, ": ", 2u);
          }
          else
          {
            *v12 = 8250;
            *(_QWORD *)(v11 + 24) += 2LL;
          }
          v13 = 16 * v6 + *(_QWORD *)(a1 + 8);
          v14 = *(_QWORD *)v13;
          if ( *(int *)(v13 + 8) < 0 )
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 40LL))(v14, a2);
          else
            sub_15537D0(v14, a2, 0, 0);
          v7 = *(_QWORD **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v7 <= 7u )
          {
            v8 = sub_16E7EE0(a2, ", align=", 8u);
          }
          else
          {
            v8 = a2;
            *v7 = 0x3D6E67696C61202CLL;
            *(_QWORD *)(a2 + 24) += 8LL;
          }
          sub_16E7AB0(v8, *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16 * v6 + 8) & 0x7FFFFFFF);
          result = *(_QWORD *)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) == result )
            break;
          ++v6;
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(a2 + 24);
          if ( (unsigned int)v5 == v6 )
            return result;
        }
        ++v6;
        result = sub_16E7EE0(a2, "\n", 1u);
      }
      while ( (unsigned int)v5 != v6 );
    }
  }
  return result;
}
