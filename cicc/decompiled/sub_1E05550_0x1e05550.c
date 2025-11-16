// Function: sub_1E05550
// Address: 0x1e05550
//
bool __fastcall sub_1E05550(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rax

  if ( a2 == a3 )
    return 1;
  v3 = sub_1E05220(a1, a3);
  v4 = sub_1E05220(a1, a2);
  return sub_1E05420(a1, v4, v3);
}
