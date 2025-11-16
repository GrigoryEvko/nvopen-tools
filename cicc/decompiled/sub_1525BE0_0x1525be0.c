// Function: sub_1525BE0
// Address: 0x1525be0
//
__int64 __fastcall sub_1525BE0(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4)
{
  unsigned int v6; // r13d
  unsigned int v7; // r8d
  __int64 v8; // rax
  __int64 result; // rax
  _DWORD v10[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = sub_153E840(a1 + 24);
  v7 = a3 - v6;
  v8 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v8 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, a4 + 16, 0, 4);
    v8 = *(unsigned int *)(a4 + 8);
    v7 = a3 - v6;
  }
  *(_DWORD *)(*(_QWORD *)a4 + 4 * v8) = v7;
  result = 0;
  ++*(_DWORD *)(a4 + 8);
  if ( v6 >= a3 )
  {
    v10[0] = sub_1524C80(a1 + 24, *a2);
    sub_1525B90(a4, v10);
    return 1;
  }
  return result;
}
