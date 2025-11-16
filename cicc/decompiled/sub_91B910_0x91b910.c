// Function: sub_91B910
// Address: 0x91b910
//
__int64 __fastcall sub_91B910(__int64 a1, _DWORD *a2)
{
  char *v2; // rax
  __int64 result; // rax

  *a2 = *((_DWORD *)qword_4F072B8 + 4 * *(int *)(a1 + 160) + 2);
  v2 = (char *)qword_4F072B8 + 16 * *(int *)(a1 + 160);
  if ( !*((_QWORD *)qword_4F073B0 + *((int *)v2 + 2)) || (result = *(_QWORD *)v2) == 0 )
    sub_91B8A0("scope for routine is NULL!", (_DWORD *)(a1 + 64), 1);
  return result;
}
