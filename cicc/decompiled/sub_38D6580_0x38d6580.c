// Function: sub_38D6580
// Address: 0x38d6580
//
__int64 __fastcall sub_38D6580(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rdx
  __int64 result; // rax

  sub_38DC4E0(a1, a2, a3);
  sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  v3 = sub_38D4B30(a1);
  if ( v3
    && *(_BYTE *)(v3 + 16) == 1
    && ((v6 = *(_QWORD *)(a1 + 264), !*(_DWORD *)(v6 + 480)) || (*(_BYTE *)(v6 + 484) & 1) == 0) )
  {
    *(_QWORD *)a2 = v3 | *(_QWORD *)a2 & 7LL;
    *(_QWORD *)(a2 + 24) = *(unsigned int *)(v3 + 72);
    result = *(_BYTE *)(a2 + 9) & 0xF3 | 4u;
    *(_BYTE *)(a2 + 9) = *(_BYTE *)(a2 + 9) & 0xF3 | 4;
  }
  else
  {
    result = *(unsigned int *)(a1 + 296);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 300) )
    {
      sub_16CD150(a1 + 288, (const void *)(a1 + 304), 0, 8, v4, v5);
      result = *(unsigned int *)(a1 + 296);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 296);
  }
  return result;
}
