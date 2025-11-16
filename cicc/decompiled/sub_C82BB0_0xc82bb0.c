// Function: sub_C82BB0
// Address: 0xc82bb0
//
int __fastcall sub_C82BB0(__int64 a1, int a2, int a3, __int64 a4, __off_t a5, __int64 a6)
{
  int result; // eax
  __int64 v8; // rdx
  _QWORD v9[2]; // [rsp+0h] [rbp-30h] BYREF
  int v10; // [rsp+10h] [rbp-20h]

  *(_QWORD *)a1 = a4;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = a3;
  result = sub_C82B10((size_t *)a1, a2, a5, a3);
  *(_DWORD *)a6 = result;
  *(_QWORD *)(a6 + 8) = v8;
  if ( result )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    v9[0] = 0;
    v9[1] = 0;
    v10 = 0;
    return sub_C82B90((__int64)v9);
  }
  return result;
}
