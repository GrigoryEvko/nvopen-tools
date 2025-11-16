// Function: sub_7BDB60
// Address: 0x7bdb60
//
__int64 __fastcall sub_7BDB60(int a1)
{
  _QWORD *v1; // rbx
  int v2; // eax
  __int64 result; // rax

  v1 = qword_4F061C0;
  v2 = *((_DWORD *)qword_4F061C0 + 2);
  if ( !v2 )
  {
    sub_7AEA70((const __m128i *)(qword_4F061C0 + 3));
    *((_DWORD *)v1 + 3) = 0;
    sub_7ADF70((__int64)(v1 + 3), 0);
    v2 = *((_DWORD *)v1 + 2);
  }
  result = (unsigned int)(v2 + 1);
  *((_DWORD *)v1 + 2) = result;
  if ( a1 )
  {
    result = (unsigned int)dword_4F0664C;
    if ( *((_DWORD *)v1 + 3) < dword_4F0664C )
    {
      sub_7AE360((__int64)(v1 + 3));
      result = (unsigned int)dword_4F0664C;
      *((_DWORD *)v1 + 3) = dword_4F0664C;
    }
  }
  return result;
}
