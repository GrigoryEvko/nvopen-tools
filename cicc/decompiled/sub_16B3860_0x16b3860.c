// Function: sub_16B3860
// Address: 0x16b3860
//
__int64 __fastcall sub_16B3860(__int64 a1, const void *a2, size_t a3)
{
  int v4; // eax
  int v5; // r14d
  unsigned int i; // r12d
  const void *v7; // rax
  __int64 v8; // rdx

  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( v4 )
  {
    v5 = v4;
    for ( i = 0; i != v5; ++i )
    {
      v7 = (const void *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, i);
      if ( a3 == v8 && (!a3 || !memcmp(v7, a2, a3)) )
        break;
    }
  }
  else
  {
    return 0;
  }
  return i;
}
