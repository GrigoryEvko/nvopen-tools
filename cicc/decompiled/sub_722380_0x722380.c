// Function: sub_722380
// Address: 0x722380
//
__int64 __fastcall sub_722380(const char *a1, __int64 a2)
{
  size_t v2; // r13
  char *v3; // rax
  __int64 v4; // rax
  __int64 result; // rax

  v2 = strlen(a1);
  v3 = sub_7222C0(*(char **)(a2 + 32));
  sub_8238A0(a2, v3);
  if ( v2 )
  {
    v4 = *(_QWORD *)(a2 + 16);
    if ( (unsigned __int64)(v4 + 1) > *(_QWORD *)(a2 + 8) )
    {
      sub_823810(a2);
      v4 = *(_QWORD *)(a2 + 16);
    }
    *(_BYTE *)(*(_QWORD *)(a2 + 32) + v4) = 46;
    ++*(_QWORD *)(a2 + 16);
    sub_8238B0(a2, a1, v2);
  }
  result = *(_QWORD *)(a2 + 16);
  if ( (unsigned __int64)(result + 1) > *(_QWORD *)(a2 + 8) )
  {
    sub_823810(a2);
    result = *(_QWORD *)(a2 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(a2 + 32) + result) = 0;
  ++*(_QWORD *)(a2 + 16);
  return result;
}
