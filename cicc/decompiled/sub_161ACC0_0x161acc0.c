// Function: sub_161ACC0
// Address: 0x161acc0
//
__int64 __fastcall sub_161ACC0(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(void); // rax

  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(__int64 (**)(void))(*(_QWORD *)v4 + 32LL);
  if ( (char *)v5 == (char *)sub_161ACB0 )
    return sub_161AC40(v4 - 160, a2, a3, a4);
  else
    return v5();
}
