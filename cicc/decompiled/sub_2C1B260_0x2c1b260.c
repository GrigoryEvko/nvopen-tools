// Function: sub_2C1B260
// Address: 0x2c1b260
//
bool __fastcall sub_2C1B260(__int64 a1)
{
  __int64 v2; // rbx
  bool result; // al
  _QWORD *v4; // rcx
  __int64 v5; // rdi
  unsigned int v6; // r14d
  __int64 v7; // r13
  bool v8; // dl
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rax

  v2 = sub_2BF04A0(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL));
  result = 0;
  if ( !v2 )
  {
    v4 = *(_QWORD **)(a1 + 48);
    if ( **(_BYTE **)(v4[1] + 40LL) == 17 )
      v2 = *(_QWORD *)(v4[1] + 40LL);
    if ( !*(_DWORD *)(a1 + 56) )
      BUG();
    v5 = *(_QWORD *)(*v4 + 40LL);
    if ( *(_BYTE *)v5 == 17 )
    {
      v6 = *(_DWORD *)(v5 + 32);
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 80) + 120LL);
      if ( v7 )
        v7 -= 24;
      v8 = v6 <= 0x40 ? *(_QWORD *)(v5 + 24) == 0 : v6 == (unsigned int)sub_C444A0(v5 + 24);
      result = v8 && v2 != 0;
      if ( result )
      {
        v9 = *(_DWORD *)(v2 + 32);
        if ( v9 <= 0x40 )
        {
          if ( *(_QWORD *)(v2 + 24) == 1 )
            goto LABEL_14;
        }
        else if ( (unsigned int)sub_C444A0(v2 + 24) == v9 - 1 )
        {
LABEL_14:
          v10 = *(_QWORD *)(a1 + 160);
          if ( !v10 )
            v10 = *(_QWORD *)(a1 + 136);
          v11 = *(_QWORD *)(v10 + 8);
          if ( !*(_DWORD *)(v7 + 56) )
            BUG();
          return *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v7 + 48) + 40LL) + 8LL) == v11;
        }
        return 0;
      }
    }
  }
  return result;
}
