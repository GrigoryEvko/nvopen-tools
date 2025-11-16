// Function: sub_DF53E0
// Address: 0xdf53e0
//
__int64 __fastcall sub_DF53E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v4; // rax
  __int64 *v5; // r12
  __int64 v6; // rsi
  __int64 *v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 *v10; // rax

  result = 0;
  if ( *(_DWORD *)(a2 + 20) - *(_DWORD *)(a2 + 24) >= (unsigned int)(*(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24)) )
  {
    v4 = *(__int64 **)(a1 + 8);
    if ( *(_BYTE *)(a1 + 28) )
      v5 = &v4[*(unsigned int *)(a1 + 20)];
    else
      v5 = &v4[*(unsigned int *)(a1 + 16)];
    if ( v4 != v5 )
    {
      while ( 1 )
      {
        v6 = *v4;
        v7 = v4;
        if ( (unsigned __int64)*v4 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v5 == ++v4 )
          return 1;
      }
      while ( v5 != v7 )
      {
        if ( *(_BYTE *)(a2 + 28) )
        {
          v8 = *(_QWORD **)(a2 + 8);
          v9 = &v8[*(unsigned int *)(a2 + 20)];
          if ( v8 == v9 )
            return 0;
          while ( *v8 != v6 )
          {
            if ( v9 == ++v8 )
              return 0;
          }
        }
        else if ( !sub_C8CA60(a2, v6) )
        {
          return 0;
        }
        v10 = v7 + 1;
        if ( v7 + 1 == v5 )
          return 1;
        while ( 1 )
        {
          v6 = *v10;
          v7 = v10;
          if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v5 == ++v10 )
            return 1;
        }
      }
    }
    return 1;
  }
  return result;
}
