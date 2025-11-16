// Function: sub_11E98F0
// Address: 0x11e98f0
//
__int64 __fastcall sub_11E98F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  int v5; // eax
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+8h] [rbp-38h]

  LODWORD(v10) = 0;
  sub_11DA4B0(a2, (int *)&v10, 1);
  v4 = *(_QWORD *)(a2 + 16);
  if ( v4 )
    return 0;
  v5 = *(_DWORD *)(a2 + 4);
  v10 = 0;
  v11 = 0;
  if ( (unsigned __int8)sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v5 & 0x7FFFFFF)), &v10, 1u) )
  {
    if ( !v11 )
    {
      v6 = *(__int64 **)(a1 + 24);
      v7 = sub_AD64C0(*(_QWORD *)(a2 + 8), 10, 0);
      v8 = sub_11CCAE0(v7, a3, v6);
      if ( v8 )
      {
        v4 = v8;
        if ( *(_BYTE *)v8 == 85 )
          *(_WORD *)(v8 + 2) = *(_WORD *)(v8 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
      }
    }
  }
  return v4;
}
