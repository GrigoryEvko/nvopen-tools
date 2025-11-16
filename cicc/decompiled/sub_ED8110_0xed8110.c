// Function: sub_ED8110
// Address: 0xed8110
//
unsigned __int64 *__fastcall sub_ED8110(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v5; // rax

  if ( (*(unsigned __int8 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F8A428) )
  {
    v3 = *a2;
    *a2 = 0;
    if ( *(_DWORD *)(v3 + 8) == 13 )
    {
      *a1 = 1;
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v3 + 8LL))(v3);
      return a1;
    }
    else
    {
      *a1 = v3 | 1;
      return a1;
    }
  }
  else
  {
    v5 = *a2;
    *a2 = 0;
    *a1 = v5 | 1;
    return a1;
  }
}
