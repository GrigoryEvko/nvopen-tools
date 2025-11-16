// Function: sub_1E73790
// Address: 0x1e73790
//
__int64 __fastcall sub_1E73790(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // r14d
  unsigned int v8; // eax
  unsigned int v9; // ebx
  int v10; // r14d
  __int64 result; // rax
  int v12; // esi
  unsigned int v13; // edx
  unsigned int v16; // [rsp+Ch] [rbp-44h]
  unsigned int v17[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v16 = *(_DWORD *)(a4 + 180);
  v7 = sub_1E72DA0(a4, *(_QWORD **)(a4 + 64), (__int64)(*(_QWORD *)(a4 + 72) - *(_QWORD *)(a4 + 64)) >> 3);
  v8 = sub_1E72DA0(a4, *(_QWORD **)(a4 + 128), (__int64)(*(_QWORD *)(a4 + 136) - *(_QWORD *)(a4 + 128)) >> 3);
  v17[0] = 0;
  v9 = v8;
  if ( v7 >= v8 )
    v9 = v7;
  if ( v9 < v16 )
    v9 = v16;
  v10 = 0;
  if ( a5 )
    v10 = sub_1E72E60(a5, v17);
  result = sub_1F4B670(*(_QWORD *)(a1 + 16));
  if ( !(_BYTE)result
    || (signed int)(v10 - *(_DWORD *)(*(_QWORD *)(a1 + 16) + 276LL) * v9) <= *(_DWORD *)(*(_QWORD *)(a1 + 16) + 276LL) )
  {
    if ( a3 )
    {
      *(_BYTE *)a2 = 1;
      result = 0;
    }
    else
    {
      result = 0;
      if ( *(_DWORD *)(a4 + 164) + v9 > *(_DWORD *)(a1 + 32) )
      {
        *(_BYTE *)a2 = 1;
        result = 0;
      }
    }
  }
  v12 = *(_DWORD *)(a4 + 276);
  v13 = v17[0];
  if ( v17[0] != v12 )
  {
    if ( *(_BYTE *)(a4 + 280) && !*(_DWORD *)(a2 + 4) )
      *(_DWORD *)(a2 + 4) = v12;
    if ( (_BYTE)result )
      *(_DWORD *)(a2 + 8) = v13;
  }
  return result;
}
