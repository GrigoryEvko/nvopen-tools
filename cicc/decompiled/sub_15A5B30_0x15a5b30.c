// Function: sub_15A5B30
// Address: 0x15a5b30
//
__int64 __fastcall sub_15A5B30(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, int *a6)
{
  __int64 v6; // rdi
  int v7; // eax
  int v9; // [rsp+8h] [rbp-8h] BYREF
  char v10; // [rsp+Ch] [rbp-4h]

  v6 = *(_QWORD *)(a1 + 8);
  if ( *((_BYTE *)a6 + 4) )
  {
    v7 = *a6;
    v10 = 1;
    v9 = v7;
  }
  else
  {
    v10 = 0;
  }
  return sub_15BD310(v6, a2, 0, 0, 0, 0, a3, a4, a5, 0, (__int64)&v9, 0, 0, 0, 1);
}
