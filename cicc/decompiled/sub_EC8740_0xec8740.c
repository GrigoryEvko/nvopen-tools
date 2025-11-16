// Function: sub_EC8740
// Address: 0xec8740
//
__int64 __fastcall sub_EC8740(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  int v3; // eax
  int v5; // eax
  int v6; // edx
  int v7; // [rsp+4h] [rbp-2Ch] BYREF
  int v8; // [rsp+8h] [rbp-28h] BYREF
  _DWORD v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v2 = sub_EC83C0(a1, &v7, &v8, "SDK");
  if ( !(_BYTE)v2 )
  {
    v3 = v7;
    *(_DWORD *)(a2 + 12) = 0;
    *(_DWORD *)a2 = v3;
    *(_QWORD *)(a2 + 4) = v8 | 0x80000000;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
    {
      v2 = sub_EC7220(a1, v9, "SDK subminor");
      if ( !(_BYTE)v2 )
      {
        v5 = v7;
        v6 = v8;
        *(_DWORD *)(a2 + 12) = 0;
        *(_DWORD *)a2 = v5;
        *(_QWORD *)(a2 + 4) = v6 & 0x7FFFFFFF | 0x8000000080000000LL | ((unsigned __int64)(v9[0] & 0x7FFFFFFF) << 32);
      }
    }
  }
  return v2;
}
