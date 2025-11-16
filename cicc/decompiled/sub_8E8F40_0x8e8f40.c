// Function: sub_8E8F40
// Address: 0x8e8f40
//
__int64 __fastcall sub_8E8F40(_BYTE *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r14
  unsigned __int8 *v5; // rax

  v2 = *a1;
  if ( *a1 == 88 )
  {
    v5 = sub_8E74B0(a1 + 1, a2);
    v3 = (__int64)v5;
    if ( *v5 == 69 )
      return (__int64)(v5 + 1);
    if ( !*(_DWORD *)(a2 + 24) )
    {
      ++*(_QWORD *)(a2 + 32);
      ++*(_QWORD *)(a2 + 48);
      *(_DWORD *)(a2 + 24) = 1;
    }
    return v3;
  }
  if ( v2 == 76 )
    return sub_8ECF80(a1, a2);
  if ( v2 != 74 && (v2 != 73 || !dword_4D0425C) )
  {
    v3 = sub_8E9FF0(a1, 0, 0, 0, 1, a2);
    sub_8EB260(a1, 0, 0, a2);
    return v3;
  }
  *(_DWORD *)(a2 + 72) = 1;
  return sub_8E9020(a1, a2);
}
