// Function: sub_2E31B00
// Address: 0x2e31b00
//
__int64 __fastcall sub_2E31B00(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax

  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == v2 )
    goto LABEL_8;
  if ( !v3 )
    BUG();
  v4 = *(_QWORD *)v3;
  v5 = *(_DWORD *)(v3 + 44);
  if ( (*(_QWORD *)v3 & 4) != 0 )
  {
    if ( (v5 & 4) != 0 )
    {
LABEL_5:
      v6 = (*(_QWORD *)(*(_QWORD *)(v3 + 16) + 24LL) >> 5) & 1LL;
      goto LABEL_6;
    }
  }
  else if ( (v5 & 4) != 0 )
  {
    while ( 1 )
    {
      v3 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      LOBYTE(v5) = *(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 44);
      if ( (v5 & 4) == 0 )
        break;
      v4 = *(_QWORD *)v3;
    }
  }
  if ( (v5 & 8) == 0 )
    goto LABEL_5;
  LOBYTE(v6) = sub_2E88A90(v3, 32, 1);
LABEL_6:
  if ( (_BYTE)v6 )
    return 0;
LABEL_8:
  if ( (unsigned __int8)sub_2E31A70(a1) )
    return 0;
  return (unsigned int)sub_2E31AC0(a1) ^ 1;
}
