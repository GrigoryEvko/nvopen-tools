// Function: sub_399E900
// Address: 0x399e900
//
void (*__fastcall sub_399E900(__int64 a1, int a2))()
{
  bool v2; // zf
  void (__fastcall **v3)(__int64, __int64); // rax
  void (*result)(); // rax

  v2 = *(_BYTE *)(a1 + 80) == 0;
  *(_DWORD *)(a1 + 76) = 1;
  v3 = *(void (__fastcall ***)(__int64, __int64))a1;
  if ( v2 )
  {
    if ( a2 > 31 )
    {
      (*v3)(a1, 144);
      return (void (*)())(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, a2);
    }
    else
    {
      return (void (*)())((__int64 (__fastcall *)(__int64, _QWORD))*v3)(a1, (unsigned __int8)(a2 + 80));
    }
  }
  else
  {
    (*v3)(a1, 144);
    result = *(void (**)())(*(_QWORD *)a1 + 8LL);
    if ( result != nullsub_1985 )
      return (void (*)())((__int64 (__fastcall *)(__int64, _QWORD))result)(a1, (unsigned int)a2);
  }
  return result;
}
