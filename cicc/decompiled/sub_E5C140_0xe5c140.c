// Function: sub_E5C140
// Address: 0xe5c140
//
__int64 __fastcall sub_E5C140(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 result; // rax

  v5 = sub_E5BD20((__int64 *)a1, a3);
  v6 = *(unsigned int *)(a1 + 368);
  if ( v6 < v5 )
    sub_C64ED0("Fragment can't be larger than a bundle size", 1u);
  v7 = *(_QWORD *)(a3 + 16);
  v8 = (unsigned int)v7 & ((_DWORD)v6 - 1);
  v9 = v8 + v5;
  if ( (*(_BYTE *)(a3 + 29) & 2) != 0 )
  {
    if ( v6 != v9 )
    {
      if ( v6 <= v9 )
        v6 = (unsigned int)(2 * v6);
      v10 = v6 - v9;
      goto LABEL_7;
    }
LABEL_16:
    result = 0;
    goto LABEL_9;
  }
  if ( v6 >= v9 || ((unsigned int)v7 & ((_DWORD)v6 - 1)) == 0 )
    goto LABEL_16;
  v10 = v6 - v8;
LABEL_7:
  if ( v10 > 0xFF )
    sub_C64ED0("Padding cannot exceed 255 bytes", 1u);
  result = (unsigned int)v10;
  v7 += v10;
LABEL_9:
  *(_BYTE *)(a3 + 30) = result;
  *(_QWORD *)(a3 + 16) = v7;
  if ( a2 && *(_BYTE *)(a2 + 28) == 1 && !*(_QWORD *)(a2 + 48) )
    *(_QWORD *)(a2 + 16) = v7;
  return result;
}
