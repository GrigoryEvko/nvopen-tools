// Function: sub_C49AB0
// Address: 0xc49ab0
//
__int64 __fastcall sub_C49AB0(__int64 a1, __int64 a2, __int64 *a3, bool *a4)
{
  const void *v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]

  v8 = *(_DWORD *)(a2 + 8);
  if ( v8 > 0x40 )
    sub_C43780((__int64)&v7, (const void **)a2);
  else
    v7 = *(const void **)a2;
  sub_C45EE0((__int64)&v7, a3);
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)a1 = v7;
  *a4 = (int)sub_C49970(a1, (unsigned __int64 *)a3) < 0;
  return a1;
}
