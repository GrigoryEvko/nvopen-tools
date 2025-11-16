// Function: sub_393A8B0
// Address: 0x393a8b0
//
__int64 *__fastcall sub_393A8B0(__int64 *a1, unsigned __int64 *a2, _DWORD **a3)
{
  _DWORD *v5; // rdi
  unsigned __int64 v7; // rax

  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F9F890) )
  {
    v5 = (_DWORD *)*a2;
    *a2 = 0;
    **a3 = v5[2];
    *a1 = 1;
    (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v5 + 8LL))(v5);
  }
  else
  {
    v7 = *a2;
    *a2 = 0;
    *a1 = v7 | 1;
  }
  return a1;
}
