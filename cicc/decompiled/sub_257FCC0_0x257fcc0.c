// Function: sub_257FCC0
// Address: 0x257fcc0
//
char __fastcall sub_257FCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char result; // al
  int v6; // [rsp-4Ch] [rbp-4Ch] BYREF
  __int64 v7[2]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v8; // [rsp-38h] [rbp-38h]
  __int64 v9; // [rsp-30h] [rbp-30h]

  result = 1;
  if ( a3 != a4 )
  {
    v7[0] = a3;
    v7[1] = a4;
    v8 = a5;
    v9 = 0;
    if ( !a5 || *(_DWORD *)(a5 + 20) == *(_DWORD *)(a5 + 24) )
      v8 = 0;
    if ( (unsigned __int8)sub_2573570(a1, v7, &v6) )
      return v6 == 1;
    else
      return sub_257EF80(a1, a2, (unsigned __int64)v7, 1);
  }
  return result;
}
