// Function: sub_9C8E80
// Address: 0x9c8e80
//
void __fastcall sub_9C8E80(__int64 a1, char **a2)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0xC00000000LL;
  if ( *((_DWORD *)a2 + 2) )
    sub_9C31C0(a1, a2);
}
