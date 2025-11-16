// Function: sub_EDA590
// Address: 0xeda590
//
__int64 *__fastcall sub_EDA590(__int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  _DWORD *v4; // r14
  unsigned __int64 v6; // rax

  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F8A428) )
  {
    v4 = (_DWORD *)*a2;
    *a2 = 0;
    **(_DWORD **)a3 = v4[2];
    sub_2240AE0(*(_QWORD *)(a3 + 8), v4 + 4);
    *a1 = 1;
    (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    v6 = *a2;
    *a2 = 0;
    *a1 = v6 | 1;
  }
  return a1;
}
