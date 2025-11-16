// Function: sub_133A2F0
// Address: 0x133a2f0
//
__int64 __fastcall sub_133A2F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // r13

  result = 1;
  if ( !(a7 | a6 | a5 | a4) )
  {
    v8 = *(_QWORD *)(a2 + 8);
    if ( v8 <= 0xFFFFFFFF && (v9 = qword_50579C0[v8]) != 0 && *(_DWORD *)(v9 + 78928) >= dword_5057900[0] )
    {
      sub_133A200(a1, v8);
      sub_1315300(a1, v9);
      sub_1339EA0(a1, v8);
      return 0;
    }
    else
    {
      return 14;
    }
  }
  return result;
}
