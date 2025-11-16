// Function: sub_5CF8B0
// Address: 0x5cf8b0
//
__int64 __fastcall sub_5CF8B0(char a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rbx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8

  v6 = sub_5CF860(a1, a2);
  result = sub_72AE80(a3);
  if ( !(_DWORD)result )
    result = sub_684B00(1909, a4);
  if ( v6 )
  {
    result = sub_73A2C0(*(_QWORD *)(v6[4] + 40), a3, v8, v9, v10);
    if ( !(_DWORD)result )
      return sub_684B00(654, a4);
  }
  return result;
}
