// Function: sub_34A8E00
// Address: 0x34a8e00
//
unsigned __int64 __fastcall sub_34A8E00(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, char a4)
{
  int v4; // r15d
  __int64 v6; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9

  v4 = a4;
  v6 = *(_QWORD *)a1;
  if ( !*(_DWORD *)(*(_QWORD *)a1 + 192LL) )
  {
    result = sub_34A32D0(
               *(_QWORD *)a1,
               (unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4),
               *(_DWORD *)(v6 + 196),
               a2,
               a3,
               a4);
    if ( (unsigned int)result <= 0xB )
    {
      *(_DWORD *)(v6 + 196) = result;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = result;
      return result;
    }
    v8 = sub_34A50A0((_QWORD *)v6, *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4));
    sub_F038C0((unsigned int *)(a1 + 8), v6 + 8, *(_DWORD *)(v6 + 196), v8, v9, v10);
  }
  return sub_34A86E0((unsigned int *)a1, a2, a3, v4);
}
