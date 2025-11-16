// Function: sub_F8F2B0
// Address: 0xf8f2b0
//
bool __fastcall sub_F8F2B0(_QWORD **a1, unsigned int a2)
{
  return *(_QWORD *)(*(_QWORD *)(**a1 - 8LL) + 32LL * *(unsigned int *)(**a1 + 72LL) + 8LL * a2) == *a1[1];
}
