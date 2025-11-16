// Function: sub_222C940
// Address: 0x222c940
//
__int64 __fastcall sub_222C940(__int64 a1, const char *a2, unsigned int a3)
{
  _QWORD *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rax

  v4 = (_QWORD *)(a1 + 104);
  if ( !sub_2207CD0((_QWORD *)(a1 + 104)) )
  {
    sub_2207CE0((__int64)v4, a2, a3);
    if ( sub_2207CD0(v4) )
    {
      sub_222BC00(a1);
      *(_DWORD *)(a1 + 120) = a3;
      *(_WORD *)(a1 + 169) = 0;
      v5 = *(_QWORD *)(a1 + 152);
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 8) = v5;
      *(_QWORD *)(a1 + 16) = v5;
      *(_QWORD *)(a1 + 24) = v5;
      v6 = *(_QWORD *)(a1 + 124);
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 132) = v6;
      *(_QWORD *)(a1 + 140) = v6;
      if ( (a3 & 2) == 0
        || (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, 0, 2, a3) != -1 )
      {
        return a1;
      }
      sub_222C7F0(a1, 0);
    }
  }
  return 0;
}
