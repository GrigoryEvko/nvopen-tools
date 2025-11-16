// Function: sub_399FD50
// Address: 0x399fd50
//
void __fastcall sub_399FD50(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rax
  __int64 v4; // [rsp-48h] [rbp-48h] BYREF
  unsigned __int64 v5; // [rsp-40h] [rbp-40h]
  char v6; // [rsp-38h] [rbp-38h]

  if ( a2 )
  {
    sub_15B1350((__int64)&v4, *(unsigned __int64 **)(a2 + 24), *(unsigned __int64 **)(a2 + 32));
    if ( v6 )
    {
      sub_15B1350((__int64)&v4, *(unsigned __int64 **)(a2 + 24), *(unsigned __int64 **)(a2 + 32));
      v2 = v5;
      v3 = *(_QWORD *)(a1 + 56);
      if ( v3 < v5 )
        sub_399EA60(a1, v5 - v3, 0);
      *(_QWORD *)(a1 + 56) = v2;
    }
  }
}
