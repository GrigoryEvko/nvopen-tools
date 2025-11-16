// Function: sub_D84370
// Address: 0xd84370
//
__int64 __fastcall sub_D84370(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // rbx
  _DWORD *v5; // rax
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  __int64 v8; // [rsp+18h] [rbp-18h]

  v5 = *(_DWORD **)(a1 + 8);
  if ( v5 && *v5 == 2 )
  {
    if ( (unsigned __int8)sub_B92100(a2, &v7) )
      return v7;
    return v4;
  }
  else if ( a3 )
  {
    return sub_FDD2C0(a3, *(_QWORD *)(a2 + 40), a4);
  }
  else
  {
    LOBYTE(v8) = 0;
    return v7;
  }
}
