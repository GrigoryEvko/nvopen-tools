// Function: sub_374D200
// Address: 0x374d200
//
__int64 __fastcall sub_374D200(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int8 v4; // dl
  __int64 v6; // rdi
  __int64 (*v7)(); // rax

  v3 = a1[5];
  if ( v3 && (unsigned __int8)sub_1056350(v3, a2) )
  {
    v6 = a1[2];
    v4 = 1;
    v7 = *(__int64 (**)())(*(_QWORD *)v6 + 560LL);
    if ( v7 != sub_2FE3120 )
      v4 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v7)(v6, a1[1], a2) ^ 1;
  }
  else
  {
    v4 = 0;
  }
  return sub_374C960((__int64)a1, *(__int64 **)(a2 + 8), v4);
}
