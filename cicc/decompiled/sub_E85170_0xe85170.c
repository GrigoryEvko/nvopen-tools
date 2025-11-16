// Function: sub_E85170
// Address: 0xe85170
//
__int64 __fastcall sub_E85170(__int64 a1, __int64 a2, int *a3)
{
  __int64 v5; // rax
  __int64 v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h]
  int v9; // [rsp+18h] [rbp-28h]

  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  if ( (unsigned __int8)sub_E81950(a3, (__int64)&v6, 0, 0) )
  {
    if ( v6 )
    {
      if ( !v7 )
      {
        v5 = *(_QWORD *)(v6 + 16);
        if ( (*(_BYTE *)(v5 + 8) & 1) == 0 || !**(_QWORD **)(v5 - 8) || v8 )
          *(_WORD *)(a2 + 12) |= 0x200u;
      }
    }
  }
  return sub_E8DCD0(a1, a2, a3);
}
