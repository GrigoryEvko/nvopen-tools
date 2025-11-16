// Function: sub_1D0DA30
// Address: 0x1d0da30
//
__int64 __fastcall sub_1D0DA30(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(void); // rdx
  __int64 result; // rax

  sub_1F010B0();
  v3 = *(_QWORD *)(a2 + 16);
  a1[77] = 0;
  a1[78] = 0;
  *a1 = &unk_49F9818;
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 128LL);
  result = 0;
  if ( v4 != sub_1D0B140 )
    result = v4();
  a1[79] = result;
  a1[80] = 0;
  a1[81] = 0;
  a1[82] = 0;
  return result;
}
