// Function: sub_2E7A200
// Address: 0x2e7a200
//
__int64 __fastcall sub_2E7A200(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  void *v4; // rdx
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // rax
  unsigned __int8 *v11; // rdi
  _QWORD *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rdx

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != result )
  {
    v4 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0xEu )
    {
      result = sub_CB6200(a2, "Constant Pool:\n", 0xFu);
    }
    else
    {
      result = 0x746E6174736E6F43LL;
      qmemcpy(v4, "Constant Pool:\n", 15);
      *(_QWORD *)(a2 + 32) += 15LL;
    }
    v5 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 4;
    if ( (_DWORD)v5 )
    {
      v6 = 0;
      do
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(a2 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 4 )
          {
            *(_DWORD *)v14 = 1885544480;
            v7 = a2;
            *(_BYTE *)(v14 + 4) = 35;
            *(_QWORD *)(a2 + 32) += 5LL;
          }
          else
          {
            v7 = sub_CB6200(a2, "  cp#", 5u);
          }
          v8 = sub_CB59D0(v7, v6);
          v9 = *(_WORD **)(v8 + 32);
          if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
          {
            sub_CB6200(v8, (unsigned __int8 *)": ", 2u);
          }
          else
          {
            *v9 = 8250;
            *(_QWORD *)(v8 + 32) += 2LL;
          }
          v10 = 16 * v6 + *(_QWORD *)(a1 + 8);
          v11 = *(unsigned __int8 **)v10;
          if ( *(_BYTE *)(v10 + 9) )
            (*(void (__fastcall **)(unsigned __int8 *, __int64))(*(_QWORD *)v11 + 48LL))(v11, a2);
          else
            sub_A5BF40(v11, a2, 0, 0);
          v12 = *(_QWORD **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 7u )
          {
            v13 = sub_CB6200(a2, ", align=", 8u);
          }
          else
          {
            v13 = a2;
            *v12 = 0x3D6E67696C61202CLL;
            *(_QWORD *)(a2 + 32) += 8LL;
          }
          sub_CB59D0(v13, 1LL << *(_BYTE *)(*(_QWORD *)(a1 + 8) + 16 * v6 + 8));
          result = *(_QWORD *)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) == result )
            break;
          ++v6;
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(a2 + 32);
          if ( (unsigned int)v5 == v6 )
            return result;
        }
        ++v6;
        result = sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
      }
      while ( (unsigned int)v5 != v6 );
    }
  }
  return result;
}
