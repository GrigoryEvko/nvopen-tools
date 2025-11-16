// Function: sub_2ECA130
// Address: 0x2eca130
//
void __fastcall sub_2ECA130(__int64 a1, unsigned __int64 **a2, char a3)
{
  __int64 *v3; // rcx
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 *v9; // rsi
  _DWORD *v10; // rax

  v3 = (__int64 *)(a2 + 5);
  v5 = *a2;
  v6 = *a2;
  if ( !a3 )
  {
    if ( !v5 )
      BUG();
    if ( (*(_BYTE *)v5 & 4) != 0 )
    {
      v6 = (unsigned __int64 *)v5[1];
      v3 = (__int64 *)(a2 + 15);
    }
    else
    {
      while ( (*((_BYTE *)v5 + 44) & 8) != 0 )
        v5 = (unsigned __int64 *)v5[1];
      v6 = (unsigned __int64 *)v5[1];
      v3 = (__int64 *)(a2 + 15);
    }
  }
  v7 = *v3;
  v8 = *v3 + 16LL * *((unsigned int *)v3 + 2);
  if ( *v3 != v8 )
  {
    while ( 1 )
    {
      if ( (*(_QWORD *)v7 & 6) != 0 || (unsigned int)(*(_DWORD *)(v7 + 8) - 1) > 0x3FFFFFFE )
        goto LABEL_8;
      v10 = (_DWORD *)(*(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL);
      if ( a3 )
      {
        if ( v10[32] <= 1u )
          goto LABEL_5;
LABEL_8:
        v7 += 16;
        if ( v8 == v7 )
          return;
      }
      else
      {
        if ( v10[12] <= 1u )
        {
LABEL_5:
          v9 = *(unsigned __int64 **)v10;
          if ( *(_WORD *)(*(_QWORD *)v10 + 68LL) == 20 || (*(_BYTE *)(v9[2] + 25) & 0x20) != 0 )
            sub_2EC62F0(*(_QWORD **)(a1 + 136), v9, v6);
          goto LABEL_8;
        }
        v7 += 16;
        if ( v8 == v7 )
          return;
      }
    }
  }
}
