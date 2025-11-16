// Function: sub_14A3230
// Address: 0x14a3230
//
__int64 __fastcall sub_14A3230(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 520LL);
  if ( (char *)v3 != (char *)sub_14A09A0 )
    return v3();
  *a3 = 0;
  return 0;
}
