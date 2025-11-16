// Function: sub_7E6F30
// Address: 0x7e6f30
//
_BOOL8 __fastcall sub_7E6F30(__int64 a1, __int64 a2, _DWORD *a3)
{
  _BOOL8 result; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  _BOOL4 v9; // [rsp+Ch] [rbp-14h] BYREF

  if ( *(_BYTE *)(a1 + 24) == 2 )
  {
    v5 = *(_QWORD *)(a1 + 56);
    result = sub_70FCE0(v5);
    if ( result )
    {
      *a3 = sub_711520(v5, a2, v6, v7, v8) == 0;
      return 1;
    }
  }
  else
  {
    v9 = 0;
    sub_7E6B40(a1, 1, 1u, a2, &v9);
    result = v9;
    if ( v9 )
    {
      *a3 = 1;
      return 1;
    }
    else
    {
      *a3 = 0;
    }
  }
  return result;
}
