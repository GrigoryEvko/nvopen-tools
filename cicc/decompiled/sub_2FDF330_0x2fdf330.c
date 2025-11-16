// Function: sub_2FDF330
// Address: 0x2fdf330
//
__int64 __fastcall sub_2FDF330(__int64 a1, __int64 a2, unsigned __int8 a3, int a4, int a5)
{
  int v7; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  v8[0] = a4;
  v7 = a5;
  if ( (a4 == -1 || a5 == -1)
    && !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _DWORD *, int *))(*(_QWORD *)a1 + 264LL))(a1, a2, v8, &v7) )
  {
    return 0;
  }
  else
  {
    return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 64LL))(a1, a2, a3);
  }
}
