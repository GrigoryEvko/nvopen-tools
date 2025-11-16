// Function: sub_B44220
// Address: 0xb44220
//
void __fastcall sub_B44220(_QWORD *a1, __int64 a2, __int64 a3)
{
  if ( !a2 )
    BUG();
  sub_B44150(a1, *(_QWORD *)(a2 + 16), (unsigned __int64 *)a2, a3);
}
