// Function: sub_2FC87D0
// Address: 0x2fc87d0
//
__int64 __fastcall sub_2FC87D0(__int64 a1, int a2)
{
  __int64 v3; // [rsp-28h] [rbp-28h] BYREF
  int v4; // [rsp-20h] [rbp-20h]

  if ( *(_WORD *)(a1 + 68) != 32 )
    return 0;
  v3 = a1;
  v4 = sub_2E88FE0(a1) + *(unsigned __int8 *)(*(_QWORD *)(a1 + 16) + 9LL);
  return sub_2FC8730(&v3, a2);
}
