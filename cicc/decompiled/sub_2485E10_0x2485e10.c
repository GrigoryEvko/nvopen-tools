// Function: sub_2485E10
// Address: 0x2485e10
//
void __fastcall sub_2485E10(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // r11
  __int64 i; // r8
  __int64 v5; // rdi
  __int64 v6; // r10
  bool v7; // cc
  unsigned int v8; // eax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx

  if ( a1 != a2 )
  {
    v2 = a1;
    v3 = a2;
    if ( a2 != a1 + 16 )
    {
      for ( i = a1 + 32; ; i += 16 )
      {
        v5 = i - 16;
        v6 = i;
        v7 = *(_DWORD *)(i - 16) <= *(_DWORD *)v2;
        if ( *(_DWORD *)(i - 16) < *(_DWORD *)v2 )
          break;
        if ( *(_DWORD *)(i - 16) == *(_DWORD *)v2 )
        {
          v8 = *(_DWORD *)(v2 + 4);
          v7 = *(_DWORD *)(i - 12) <= v8;
          if ( *(_DWORD *)(i - 12) < v8 )
            break;
        }
        if ( v7 )
        {
          v9 = *(_QWORD *)(i - 8);
          if ( v9 < *(_QWORD *)(v2 + 8) )
            goto LABEL_12;
        }
        sub_2485DB0((unsigned int *)v5);
LABEL_8:
        if ( v3 == v6 )
          return;
      }
      v9 = *(_QWORD *)(i - 8);
LABEL_12:
      v10 = *(_QWORD *)(i - 16);
      v11 = (v5 - v2) >> 4;
      if ( v5 - v2 > 0 )
      {
        do
        {
          v12 = *(_QWORD *)(v5 - 16);
          v5 -= 16;
          *(_QWORD *)(v5 + 16) = v12;
          *(_QWORD *)(v5 + 24) = *(_QWORD *)(v5 + 8);
          --v11;
        }
        while ( v11 );
      }
      *(_QWORD *)v2 = v10;
      *(_QWORD *)(v2 + 8) = v9;
      goto LABEL_8;
    }
  }
}
