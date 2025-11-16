// Function: sub_1636880
// Address: 0x1636880
//
__int64 __fastcall sub_1636880(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 (*v5)(); // rax

  v3 = sub_15E0530(a2);
  v4 = sub_16033D0(v3);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 24LL);
  if ( v5 == sub_1635ED0 || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v5)(v4, a1, a2) )
    return sub_1560180(a2 + 112, 35);
  else
    return 1;
}
