// Function: sub_DFB3A0
// Address: 0xdfb3a0
//
__int64 __fastcall sub_DFB3A0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1080LL);
  if ( (char *)v3 != (char *)sub_DF6090 )
    return v3();
  *a3 = 0;
  return 0;
}
