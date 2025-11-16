// Function: sub_620F30
// Address: 0x620f30
//
__int64 __fastcall sub_620F30(__int16 *a1, int a2, _DWORD *a3)
{
  __int64 result; // rax
  int v5; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  *a3 = 0;
  sub_620E00(a1, a2, v6, &v5);
  if ( !v5 && (!a2 || *a1 >= 0) )
    return v6[0];
  result = v6[0];
  *a3 = 1;
  return result;
}
