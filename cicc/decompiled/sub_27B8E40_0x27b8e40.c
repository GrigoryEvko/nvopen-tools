// Function: sub_27B8E40
// Address: 0x27b8e40
//
__int64 __fastcall sub_27B8E40(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // [rsp+0h] [rbp-30h] BYREF
  __int64 v4; // [rsp+8h] [rbp-28h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h] BYREF
  __int64 v6; // [rsp+18h] [rbp-18h] BYREF

  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
        return *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    }
  }
  if ( (unsigned __int8)sub_D22AF0(a1, &v3, &v4, &v5, &v6) )
    return v3;
  return *(_QWORD *)(a1 - 96);
}
