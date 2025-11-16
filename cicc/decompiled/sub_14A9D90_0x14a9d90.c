// Function: sub_14A9D90
// Address: 0x14a9d90
//
void __fastcall sub_14A9D90(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // cl
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax

  if ( (_DWORD)a2 != (_DWORD)a3 )
  {
    if ( (unsigned int)a2 > 0x3F || (unsigned int)a3 > 0x40 )
    {
      sub_16A5260(a1, a2, a3);
    }
    else
    {
      v3 = a2 - a3;
      v4 = *(_QWORD *)a1;
      v5 = 0xFFFFFFFFFFFFFFFFLL >> (v3 + 64) << a2;
      if ( *(_DWORD *)(a1 + 8) > 0x40u )
        *(_QWORD *)v4 |= v5;
      else
        *(_QWORD *)a1 = v4 | v5;
    }
  }
}
