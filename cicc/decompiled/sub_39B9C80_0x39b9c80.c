// Function: sub_39B9C80
// Address: 0x39b9c80
//
__int64 __fastcall sub_39B9C80(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rdx
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v6 = sub_1560340((_QWORD *)(*(_QWORD *)a2 + 112LL), -1, "no-frame-pointer-elim", 0x15u);
  v2 = (_DWORD *)sub_155D8B0(&v6);
  v3 = 0;
  if ( v4 == 4 )
    LOBYTE(v3) = *v2 == 1702195828;
  return v3;
}
