// Function: sub_1DD9680
// Address: 0x1dd9680
//
void __fastcall sub_1DD9680(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  unsigned __int64 v5; // rbx
  __int16 v6; // ax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx

  v3 = (_QWORD *)(a1 + 24);
  if ( a1 + 24 != *(_QWORD *)(a1 + 32) )
  {
    do
    {
      v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = *(_WORD *)(v5 + 46);
      v3 = (_QWORD *)v5;
      if ( (v6 & 4) != 0 || (v6 & 8) == 0 )
        v7 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL) >> 6) & 1LL;
      else
        LOBYTE(v7) = sub_1E15D00(v5, 64, 1);
      if ( !(_BYTE)v7 )
        break;
      v8 = *(unsigned int *)(v5 + 40);
      if ( (_DWORD)v8 )
      {
        v9 = 5 * v8;
        v10 = 0;
        v11 = 8 * v9;
        do
        {
          while ( 1 )
          {
            v12 = v10 + *(_QWORD *)(v5 + 32);
            if ( *(_BYTE *)v12 == 4 && a2 == *(_QWORD *)(v12 + 24) )
              break;
            v10 += 40;
            if ( v10 == v11 )
              goto LABEL_12;
          }
          v10 += 40;
          *(_QWORD *)(v12 + 24) = a3;
        }
        while ( v10 != v11 );
      }
LABEL_12:
      ;
    }
    while ( v5 != *(_QWORD *)(a1 + 32) );
  }
  sub_1DD9570(a1, a2, a3);
}
