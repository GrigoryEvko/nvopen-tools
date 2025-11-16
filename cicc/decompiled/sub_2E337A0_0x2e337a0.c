// Function: sub_2E337A0
// Address: 0x2e337a0
//
void __fastcall sub_2E337A0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  unsigned __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 i; // rdx

  v3 = (_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) != a1 + 48 )
  {
    do
    {
      v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = *(_DWORD *)(v5 + 44);
      v3 = (_QWORD *)v5;
      if ( (v6 & 4) != 0 || (v6 & 8) == 0 )
        v7 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) >> 9) & 1LL;
      else
        LOBYTE(v7) = sub_2E88A90(v5, 512, 1);
      if ( !(_BYTE)v7 )
        break;
      v8 = *(_QWORD *)(v5 + 32);
      for ( i = v8 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF); i != v8; v8 += 40 )
      {
        while ( *(_BYTE *)v8 != 4 || a2 != *(_QWORD *)(v8 + 24) )
        {
          v8 += 40;
          if ( i == v8 )
            goto LABEL_12;
        }
        *(_QWORD *)(v8 + 24) = a3;
      }
LABEL_12:
      ;
    }
    while ( *(_QWORD *)(a1 + 56) != v5 );
  }
  sub_2E33690(a1, a2, a3);
}
