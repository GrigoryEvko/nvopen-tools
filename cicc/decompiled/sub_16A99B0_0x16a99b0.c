// Function: sub_16A99B0
// Address: 0x16a99b0
//
__int64 __fastcall sub_16A99B0(__int64 a1, __int64 a2, __int64 *a3, bool *a4)
{
  const void *v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]

  v8 = *(_DWORD *)(a2 + 8);
  if ( v8 > 0x40 )
    sub_16A4FD0((__int64)&v7, (const void **)a2);
  else
    v7 = *(const void **)a2;
  sub_16A7200((__int64)&v7, a3);
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)a1 = v7;
  *a4 = (int)sub_16A9900(a1, (unsigned __int64 *)a3) < 0;
  return a1;
}
