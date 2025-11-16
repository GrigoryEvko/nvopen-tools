// Function: sub_21BE110
// Address: 0x21be110
//
__int64 __fastcall sub_21BE110(__int64 a1)
{
  __int64 v1; // rdi
  __int64 (*v2)(void); // rax
  __int64 v3; // rdi
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 480);
  v2 = *(__int64 (**)(void))(*(_QWORD *)v1 + 56LL);
  if ( (char *)v2 == (char *)sub_214ABA0 )
    v3 = v1 + 696;
  else
    v3 = v2();
  result = *(unsigned int *)(*(_QWORD *)(v3 + 81552) + 82312LL);
  if ( (_DWORD)result == -1 )
    return 2 * (unsigned int)((*(_BYTE *)(*(_QWORD *)(v3 + 8) + 792LL) & 2) == 0);
  return result;
}
