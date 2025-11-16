// Function: sub_2BDFD50
// Address: 0x2bdfd50
//
unsigned __int64 __fastcall sub_2BDFD50(__int64 a1, __int64 a2, __int64 a3, int a4, _QWORD *a5)
{
  const char *v5; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned __int64 result; // rax

  *(_QWORD *)a1 = 0x160000005ELL;
  *(_QWORD *)(a1 + 8) = 0x1700000024LL;
  *(_QWORD *)(a1 + 24) = 0x140000002ALL;
  *(_QWORD *)(a1 + 32) = 0x150000002BLL;
  *(_QWORD *)(a1 + 40) = 0x120000003FLL;
  *(_QWORD *)(a1 + 48) = 0x130000007CLL;
  *(_QWORD *)(a1 + 56) = 0x130000000ALL;
  *(_QWORD *)(a1 + 64) = 0x1300000000LL;
  *(_QWORD *)(a1 + 72) = 0xA6E0C6608620030LL;
  *(_QWORD *)(a1 + 80) = 0xB7609740D72LL;
  *(_QWORD *)(a1 + 88) = 0x7615C5C2F2F2222LL;
  *(_QWORD *)(a1 + 96) = 0xD720A6E0C660862LL;
  *(_WORD *)(a1 + 108) = 0;
  v5 = ".[\\*^$";
  *(_QWORD *)(a1 + 16) = 46;
  *(_DWORD *)(a1 + 104) = 192285044;
  *(_QWORD *)(a1 + 112) = "^$\\.*+?()[]{}|";
  *(_QWORD *)(a1 + 120) = ".[\\*^$";
  *(_QWORD *)(a1 + 128) = ".[\\()*+?{|^$";
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 140) = a4;
  if ( (a4 & 0x10) != 0 )
  {
    *(_QWORD *)(a1 + 152) = a1 + 72;
    v5 = "^$\\.*+?()[]{}|";
  }
  else
  {
    *(_QWORD *)(a1 + 152) = a1 + 88;
    if ( (a4 & 0x20) == 0 )
    {
      v5 = ".[\\()*+?{|^$";
      if ( (a4 & 0x40) == 0 )
      {
        v5 = ".[\\*^$\n";
        if ( (a4 & 0x100) == 0 )
        {
          v5 = ".[\\()*+?{|^$\n";
          if ( (a4 & 0x200) == 0 )
          {
            v5 = 0;
            if ( (a4 & 0x80) != 0 )
              v5 = ".[\\()*+?{|^$";
          }
        }
      }
    }
  }
  *(_QWORD *)(a1 + 176) = a2;
  *(_QWORD *)(a1 + 160) = v5;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = a3;
  v7 = sub_22091A0(&qword_4FD6808);
  v8 = *(_QWORD *)(*a5 + 8LL);
  if ( v7 >= *(_QWORD *)(*a5 + 16LL) || (v9 = *(_QWORD *)(v8 + 8 * v7)) == 0 )
    sub_426219(&qword_4FD6808, ".[\\()*+?{|^$", *a5, v8);
  *(_QWORD *)(a1 + 192) = v9;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0;
  *(_BYTE *)(a1 + 216) = 0;
  if ( (*(_BYTE *)(a1 + 140) & 0x10) != 0 )
  {
    *(_QWORD *)(a1 + 240) = 0;
    *(_QWORD *)(a1 + 232) = sub_2BDDFD0;
  }
  else
  {
    *(_QWORD *)(a1 + 240) = 0;
    *(_QWORD *)(a1 + 232) = sub_2BDF300;
  }
  result = *(_QWORD *)(a1 + 184);
  if ( *(_QWORD *)(a1 + 176) == result )
  {
    *(_DWORD *)(a1 + 144) = 27;
  }
  else
  {
    result = *(unsigned int *)(a1 + 136);
    if ( (_DWORD)result )
    {
      if ( (_DWORD)result == 2 )
      {
        return (unsigned __int64)sub_2BDFB70(a1);
      }
      else if ( (_DWORD)result == 1 )
      {
        return (unsigned __int64)sub_2BDF830(a1);
      }
    }
    else
    {
      return sub_2BDF460((unsigned __int8 *)a1);
    }
  }
  return result;
}
