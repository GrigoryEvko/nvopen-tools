// Function: sub_37F9090
// Address: 0x37f9090
//
__int64 __fastcall sub_37F9090(__int64 a1)
{
  int v1; // eax
  __int64 v2; // r8
  __int64 result; // rax
  const char *v4; // [rsp+0h] [rbp-30h] BYREF
  int v5; // [rsp+10h] [rbp-20h]
  __int16 v6; // [rsp+20h] [rbp-10h]

  v1 = *(_DWORD *)(a1 + 336);
  v2 = *(_QWORD *)(a1 + 24);
  v4 = "__ehinfo.";
  v5 = v1;
  v6 = 2307;
  result = sub_E6C460(v2, &v4);
  *(_WORD *)(result + 12) |= 1u;
  return result;
}
