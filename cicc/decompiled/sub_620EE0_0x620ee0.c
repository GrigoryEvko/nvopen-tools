// Function: sub_620EE0
// Address: 0x620ee0
//
__int64 __fastcall sub_620EE0(_WORD *a1, int a2, _DWORD *a3)
{
  __int64 result; // rax
  int v5; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v6[3]; // [rsp+8h] [rbp-18h] BYREF

  *a3 = 0;
  sub_620E00(a1, a2, v6, &v5);
  result = v6[0];
  if ( !a2 && v6[0] < 0 || v5 )
    *a3 = 1;
  return result;
}
