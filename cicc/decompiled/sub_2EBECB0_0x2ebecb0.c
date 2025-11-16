// Function: sub_2EBECB0
// Address: 0x2ebecb0
//
void __fastcall sub_2EBECB0(_QWORD *a1, int a2, unsigned int a3)
{
  _QWORD *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rdi

  v4 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  if ( a2 < 0 )
    v5 = *(_QWORD *)(a1[7] + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v5 = *(_QWORD *)(a1[38] + 8LL * (unsigned int)a2);
  while ( v5 )
  {
    while ( 1 )
    {
      v6 = v5;
      v5 = *(_QWORD *)(v5 + 32);
      if ( a3 - 1 > 0x3FFFFFFE )
        break;
      sub_2EAB1E0(v6, a3, v4);
      if ( !v5 )
        return;
    }
    sub_2EAB0C0(v6, a3);
  }
}
