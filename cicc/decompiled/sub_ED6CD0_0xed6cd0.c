// Function: sub_ED6CD0
// Address: 0xed6cd0
//
__int64 __fastcall sub_ED6CD0(__int64 a1)
{
  _QWORD *v1; // rdi
  __int64 (*v2)(void); // rax

  v1 = *(_QWORD **)(a1 + 128);
  v2 = *(__int64 (**)(void))(*v1 + 96LL);
  if ( (char *)v2 == (char *)sub_ED6070 )
    return (v1[6] >> 60) & 1LL;
  else
    return v2();
}
