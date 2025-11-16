// Function: sub_851E50
// Address: 0x851e50
//
size_t __fastcall sub_851E50(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  size_t result; // rax
  __time_t v4[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( a1 )
  {
    v1 = a1;
    do
    {
      if ( (*(_BYTE *)(v1 + 72) & 4) != 0 )
      {
        v2 = *(_QWORD *)(v1 + 8);
        if ( v2 )
        {
          sub_723E40(v2, v4);
          sub_851CB0(*(const char **)(v1 + 8));
          result = fwrite(v4, 8u, 1u, qword_4F5FB40);
        }
      }
      if ( *(_QWORD *)(v1 + 40) )
        result = sub_851E50();
      v1 = *(_QWORD *)(v1 + 56);
    }
    while ( v1 );
  }
  return result;
}
