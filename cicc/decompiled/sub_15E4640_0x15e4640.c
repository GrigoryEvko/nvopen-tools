// Function: sub_15E4640
// Address: 0x15e4640
//
__int64 __fastcall sub_15E4640(__int64 a1)
{
  _DWORD *v1; // rax
  unsigned int v2; // r8d
  __int64 v3; // rdx
  __int64 v5; // [rsp+8h] [rbp-8h] BYREF

  v5 = sub_1560340((_QWORD *)(a1 + 112), -1, "null-pointer-is-valid", 0x15u);
  v1 = (_DWORD *)sub_155D8B0(&v5);
  v2 = 0;
  if ( v3 == 4 )
    LOBYTE(v2) = *v1 == 1702195828;
  return v2;
}
