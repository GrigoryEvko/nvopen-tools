// Function: sub_21BE190
// Address: 0x21be190
//
__int64 __fastcall sub_21BE190(__int64 a1)
{
  __int64 v2; // rdi
  __int64 (*v3)(void); // rax
  __int64 v4; // rdi

  v2 = *(_QWORD *)(a1 + 480);
  v3 = *(__int64 (**)(void))(*(_QWORD *)v2 + 56LL);
  if ( (char *)v3 == (char *)sub_214ABA0 )
    v4 = v2 + 696;
  else
    v4 = v3();
  return sub_21CF380(v4, *(_QWORD *)(a1 + 256));
}
