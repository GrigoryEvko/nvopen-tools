// Function: sub_2EBE500
// Address: 0x2ebe500
//
__int64 __fastcall sub_2EBE500(__int64 a1, int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r12

  if ( a3 == a4 )
    return a3;
  v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)a1 + 16LL));
  v10 = sub_2FF6970(v8, a3, a4, v9);
  v11 = v10;
  if ( v10 && a3 != v10 )
  {
    if ( a5 > *(unsigned __int16 *)(*(_QWORD *)v10 + 20LL) )
      return 0;
    else
      sub_2EBE4E0(a1, a2, v10);
  }
  return v11;
}
