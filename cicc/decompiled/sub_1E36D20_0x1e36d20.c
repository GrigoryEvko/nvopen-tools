// Function: sub_1E36D20
// Address: 0x1e36d20
//
__int64 __fastcall sub_1E36D20(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdx
  __int64 result; // rax
  bool v5; // zf
  __int64 v6; // [rsp-28h] [rbp-28h] BYREF
  char v7; // [rsp-20h] [rbp-20h]
  __int64 v8; // [rsp-8h] [rbp-8h]

  v3 = *(_QWORD *)(a2 + 464);
  if ( v3 )
  {
    v8 = v2;
    result = sub_1E36CE0((__int64)&v6, a1, v3);
    if ( v7 )
    {
      result = v6;
      v5 = *(_BYTE *)(a2 + 80) == 0;
      *(_QWORD *)(a2 + 72) = v6;
      if ( v5 )
        *(_BYTE *)(a2 + 80) = 1;
    }
    else if ( *(_BYTE *)(a2 + 80) )
    {
      *(_BYTE *)(a2 + 80) = 0;
    }
  }
  return result;
}
