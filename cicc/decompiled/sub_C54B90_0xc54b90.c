// Function: sub_C54B90
// Address: 0xc54b90
//
__int64 __fastcall sub_C54B90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rdx

  v2 = *(_QWORD *)(a2 + 32);
  if ( v2 == 1 )
    v3 = qword_4C5C728 + 6;
  else
    v3 = v2 + qword_4C5C718 + 5;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( v4 )
  {
    if ( *(_QWORD *)(a2 + 64) )
      v4 = *(_QWORD *)(a2 + 64);
    v3 += v4 + (-(__int64)(((*(_BYTE *)(a2 + 13) >> 1) & 2) == 0) & 0xFFFFFFFFFFFFFFFDLL) + 6;
  }
  return v3;
}
