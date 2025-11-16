// Function: sub_331C500
// Address: 0x331c500
//
unsigned __int64 __fastcall sub_331C500(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9

  v4 = *(_QWORD *)a1;
  if ( !*(_DWORD *)(*(_QWORD *)a1 + 136LL) )
  {
    result = sub_325EF20(
               *(_QWORD *)a1,
               (unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4),
               *(_DWORD *)(v4 + 140),
               a2,
               a3);
    if ( (unsigned int)result <= 8 )
    {
      *(_DWORD *)(v4 + 140) = result;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = result;
      return result;
    }
    v6 = sub_32AEF50((_QWORD *)v4, *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4));
    sub_F038C0((unsigned int *)(a1 + 8), v4 + 8, *(_DWORD *)(v4 + 140), v6, v7, v8);
  }
  return sub_331BE70((unsigned int *)a1, a2, a3);
}
