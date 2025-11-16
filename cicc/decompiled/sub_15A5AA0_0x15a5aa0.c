// Function: sub_15A5AA0
// Address: 0x15a5aa0
//
__int64 __fastcall sub_15A5AA0(__int64 a1, __int64 a2, __int64 a3, int a4, int *a5, __int64 a6, __int64 a7, __int64 a8)
{
  __int64 v10; // r14
  int v11; // eax
  int v12; // edx
  int v14; // [rsp+8h] [rbp-28h] BYREF
  char v15; // [rsp+Ch] [rbp-24h]

  v10 = *(_QWORD *)(a1 + 8);
  if ( *((_BYTE *)a5 + 4) )
  {
    v11 = *a5;
    v15 = 1;
    v14 = v11;
  }
  else
  {
    v15 = 0;
  }
  v12 = 0;
  if ( a8 )
    v12 = sub_161FF10(v10, a7, a8);
  return sub_15BD310(v10, 15, v12, 0, 0, 0, a2, a3, a4, 0, (__int64)&v14, 0, 0, 0, 1);
}
