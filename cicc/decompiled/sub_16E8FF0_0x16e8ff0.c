// Function: sub_16E8FF0
// Address: 0x16e8ff0
//
__int64 __fastcall sub_16E8FF0(__int64 a1)
{
  const char *v1; // rax
  unsigned __int64 v2; // rcx
  void *v3; // rdx
  __int64 result; // rax
  const char *v5; // rdx
  unsigned __int64 v6; // rcx

  v1 = *(const char **)a1;
  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 >= v2 )
  {
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 7;
    v1 = (const char *)&unk_4FA17D0;
    *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
    v3 = &unk_4FA17D1;
    goto LABEL_5;
  }
  v3 = (void *)(v1 + 1);
  if ( v2 <= (unsigned __int64)(v1 + 1) || *v1 != 91 || v1[1] != 46 )
  {
LABEL_5:
    *(_QWORD *)a1 = v3;
    return *(unsigned __int8 *)v1;
  }
  *(_QWORD *)a1 = v1 + 2;
  result = sub_16E8EC0((const char **)a1, 46);
  v5 = *(const char **)a1;
  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 < v6 && v6 > (unsigned __int64)(v5 + 1) && *v5 == 46 && v5[1] == 93 )
  {
    *(_QWORD *)a1 = v5 + 2;
  }
  else
  {
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 3;
    *(_QWORD *)a1 = &unk_4FA17D0;
    *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
  }
  return result;
}
