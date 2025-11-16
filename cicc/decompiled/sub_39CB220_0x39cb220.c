// Function: sub_39CB220
// Address: 0x39cb220
//
void __fastcall sub_39CB220(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rcx
  unsigned __int64 v7[2]; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v8[64]; // [rsp+10h] [rbp-40h] BYREF

  v6 = *(unsigned int *)(a3 + 8);
  if ( (_DWORD)v6 != 1 && *(_BYTE *)(a1[25] + 4501) )
  {
    v7[0] = (unsigned __int64)v8;
    v7[1] = 0x200000000LL;
    if ( (_DWORD)v6 )
      sub_39C75B0((__int64)v7, (char **)a3, a3, v6, a5, a6);
    sub_39C7DB0(a1, a2, (__int64)v7);
    if ( (_BYTE *)v7[0] != v8 )
      _libc_free(v7[0]);
  }
  else
  {
    sub_39CB010(a1, a2, **(_QWORD **)a3, *(_QWORD *)(*(_QWORD *)a3 + 16 * v6 - 8));
  }
}
