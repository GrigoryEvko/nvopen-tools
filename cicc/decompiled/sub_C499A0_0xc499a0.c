// Function: sub_C499A0
// Address: 0xc499a0
//
__int64 __fastcall sub_C499A0(__int64 a1, __int64 a2, __int64 *a3, bool *a4)
{
  int v6; // eax
  bool v7; // zf
  bool v8; // sf
  const void *v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]

  v11 = *(_DWORD *)(a2 + 8);
  if ( v11 > 0x40 )
    sub_C43780((__int64)&v10, (const void **)a2);
  else
    v10 = *(const void **)a2;
  sub_C46B40((__int64)&v10, a3);
  *(_DWORD *)(a1 + 8) = v11;
  *(_QWORD *)a1 = v10;
  v6 = sub_C49970(a1, (unsigned __int64 *)a2);
  v7 = v6 == 0;
  v8 = v6 < 0;
  *a4 = !v8 && !v7;
  return a1;
}
