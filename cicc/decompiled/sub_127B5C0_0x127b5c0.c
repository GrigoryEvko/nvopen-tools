// Function: sub_127B5C0
// Address: 0x127b5c0
//
__int64 __fastcall sub_127B5C0(__int64 a1, _DWORD *a2)
{
  char *v2; // rax
  __int64 result; // rax

  *a2 = *((_DWORD *)qword_4F072B8 + 4 * *(int *)(a1 + 160) + 2);
  v2 = (char *)qword_4F072B8 + 16 * *(int *)(a1 + 160);
  if ( !*((_QWORD *)qword_4F073B0 + *((int *)v2 + 2)) || (result = *(_QWORD *)v2) == 0 )
    sub_127B550("scope for routine is NULL!", (_DWORD *)(a1 + 64), 1);
  return result;
}
