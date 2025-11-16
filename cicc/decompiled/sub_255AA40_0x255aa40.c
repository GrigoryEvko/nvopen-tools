// Function: sub_255AA40
// Address: 0x255aa40
//
__int64 __fastcall sub_255AA40(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  int v7; // eax
  unsigned int v8; // esi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  _DWORD v17[9]; // [rsp+Ch] [rbp-24h] BYREF

  result = sub_2509800((_QWORD *)(a1 + 72));
  if ( (_BYTE)result == 4 )
  {
    v7 = *(_DWORD *)(a1 + 100);
    if ( (unsigned __int8)v7 != 255 && (*(_DWORD *)(a1 + 100) & 0xFC) != 0xFC )
    {
      if ( (*(_DWORD *)(a1 + 100) & 0xDC) == 0xDC )
      {
        v8 = 12;
      }
      else
      {
        v8 = 3;
        if ( (*(_DWORD *)(a1 + 100) & 0xEC) != 0xEC )
        {
          result = (unsigned __int8)v7 & 0xCC;
          if ( (_DWORD)result != 204 )
            return result;
          sub_255AA10(v17, 0);
          v8 = v17[0] | 0xF;
        }
      }
      v9 = sub_A77AB0(a3, v8);
      return sub_255A480(a4, v9, v10, v11, v12, v13);
    }
    v14 = sub_A77AB0(a3, 0);
    result = *(unsigned int *)(a4 + 8);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), result + 1, 8u, v15, v16);
      result = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = v14;
    ++*(_DWORD *)(a4 + 8);
  }
  return result;
}
