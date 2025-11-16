// Function: sub_3103560
// Address: 0x3103560
//
char __fastcall sub_3103560(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  char result; // al
  __int64 v8; // rax

  v6 = *(_QWORD *)(a2 + 40);
  if ( v6 != **(_QWORD **)(a4 + 32) )
    return sub_3103540((unsigned __int8 (__fastcall ***)(_QWORD, __int64))a1, a4, v6, a3, v6, a6);
  result = 1;
  if ( *(_BYTE *)(a1 + 41) )
  {
    v8 = sub_AA5030(v6, 1);
    return v8 && a2 == v8 - 24;
  }
  return result;
}
