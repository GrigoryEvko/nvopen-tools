// Function: sub_72DB50
// Address: 0x72db50
//
__int64 *__fastcall sub_72DB50(__int64 a1, char a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax

  v3 = *(_QWORD *)(a1 + 48);
  if ( !*(_DWORD *)(v3 + 160) )
    return 0;
  v4 = sub_72B840(v3);
  return sub_72DB00(a1, a2, v4);
}
