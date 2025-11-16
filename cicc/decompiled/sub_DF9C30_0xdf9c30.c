// Function: sub_DF9C30
// Address: 0xdf9c30
//
__int64 __fastcall sub_DF9C30(__int64 *a1, _BYTE *a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(void); // rax

  v2 = *a1;
  v3 = *(__int64 (**)(void))(*(_QWORD *)v2 + 344LL);
  if ( (char *)v3 == (char *)sub_DF8580 )
    return sub_DF7D80(v2 + 8, a2);
  else
    return v3();
}
