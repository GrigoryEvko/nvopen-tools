// Function: sub_14A2FC0
// Address: 0x14a2fc0
//
__int64 __fastcall sub_14A2FC0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 416LL);
  if ( v1 == sub_14A0930 )
    return 0;
  else
    return v1();
}
