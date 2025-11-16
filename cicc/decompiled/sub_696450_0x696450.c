// Function: sub_696450
// Address: 0x696450
//
_BOOL8 __fastcall sub_696450(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  _BOOL8 result; // rax
  int v4; // eax
  int v5; // [rsp+Ch] [rbp-34h] BYREF
  _BYTE v6[48]; // [rsp+10h] [rbp-30h] BYREF

  if ( dword_4F077C4 == 2 )
  {
    v4 = sub_836C50(0, a1, a2, 1, 1, 1, 0, 0, 0, (__int64)v6, 0, (__int64)&v5, 0);
    return (v5 | v4) != 0;
  }
  else
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
    result = 1;
    if ( v2 != a2 )
      return (unsigned int)sub_8DED30(v2, a2, 3) != 0;
  }
  return result;
}
