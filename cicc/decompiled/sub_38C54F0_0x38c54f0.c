// Function: sub_38C54F0
// Address: 0x38c54f0
//
__int64 __fastcall sub_38C54F0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  char v5; // r13
  _BYTE *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r8

  v5 = a3;
  v6 = *(_BYTE **)(a1[1] + 16);
  (*(void (__fastcall **)(_BYTE *, __int64, __int64, __int64 *))(*(_QWORD *)v6 + 40LL))(v6, a2, a3, a1);
  v7 = (unsigned int)sub_38C54A0(a1[1], v5);
  if ( v6[357] && a4 )
    return sub_38C4F40(a1, v8, v7);
  else
    return sub_38DDD30(a1, v8, v7, 0);
}
