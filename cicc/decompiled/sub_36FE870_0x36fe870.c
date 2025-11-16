// Function: sub_36FE870
// Address: 0x36fe870
//
__int64 __fastcall sub_36FE870(__int64 a1)
{
  __int64 *v1; // rdi
  __int64 result; // rax

  sub_36FFE70(*(_QWORD *)(a1 + 240));
  v1 = **(__int64 ***)(a1 + 240);
  result = v1[2];
  if ( v1[4] != result )
    return sub_CB5AE0(v1);
  return result;
}
