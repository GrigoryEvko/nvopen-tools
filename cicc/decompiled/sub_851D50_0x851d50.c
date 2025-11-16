// Function: sub_851D50
// Address: 0x851d50
//
size_t __fastcall sub_851D50(__int64 *a1)
{
  __int64 *v1; // rbx
  int v2; // eax
  _DWORD ptr[3]; // [rsp+Ch] [rbp-14h] BYREF

  if ( a1 )
  {
    v1 = a1;
    do
    {
      fwrite(v1 + 1, 4u, 1u, qword_4F5FB40);
      v2 = *((_DWORD *)v1 + 2);
      if ( v2 == 1 )
      {
        fwrite((char *)v1 + 12, 4u, 1u, qword_4F5FB40);
        fwrite(v1 + 2, 1u, 1u, qword_4F5FB40);
      }
      else
      {
        if ( v2 != 2 )
          sub_721090();
        fwrite((char *)v1 + 12, 4u, 1u, qword_4F5FB40);
      }
      sub_851CB0((const char *)v1[3]);
      fwrite(v1 + 4, 8u, 1u, qword_4F5FB40);
      v1 = (__int64 *)*v1;
    }
    while ( v1 );
  }
  ptr[0] = 0;
  return fwrite(ptr, 4u, 1u, qword_4F5FB40);
}
