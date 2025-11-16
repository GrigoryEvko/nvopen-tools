// Function: sub_7F73A0
// Address: 0x7f73a0
//
__int64 __fastcall sub_7F73A0(__int64 a1, const __m128i *a2, int *a3)
{
  __int64 v3; // r13
  __int64 v5; // rdi
  __int64 result; // rax
  _BYTE v8[4]; // [rsp+4h] [rbp-2Ch] BYREF
  int v9; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v10[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v9 = 0;
  v10[0] = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(v3 + 16);
    if ( !v5 )
    {
      if ( (*(_BYTE *)a1 & 2) != 0 )
        return result;
      goto LABEL_6;
    }
    result = sub_87ADD0(v5, v8, &v9, v10);
    if ( (*(_BYTE *)a1 & 2) == 0 && !v9 )
    {
      result = v10[0];
      if ( !v10[0] )
      {
LABEL_6:
        if ( *(char *)(v3 + 49) >= 0 )
          return sub_7F72E0(v3, a2, 1, (__int64)qword_4D03F68, a3);
      }
    }
  }
  return result;
}
