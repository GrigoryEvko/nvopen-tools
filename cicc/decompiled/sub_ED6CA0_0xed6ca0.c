// Function: sub_ED6CA0
// Address: 0xed6ca0
//
__int64 __fastcall sub_ED6CA0(__int64 a1)
{
  _QWORD *v1; // rdi
  __int64 (*v2)(void); // rax

  v1 = *(_QWORD **)(a1 + 128);
  v2 = *(__int64 (**)(void))(*v1 + 88LL);
  if ( (char *)v2 == (char *)sub_ED6060 )
    return (v1[6] >> 55) & 1LL;
  else
    return v2();
}
