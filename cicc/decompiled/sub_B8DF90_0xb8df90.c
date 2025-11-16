// Function: sub_B8DF90
// Address: 0xb8df90
//
void __fastcall sub_B8DF90(__int64 a1, _BYTE *a2)
{
  _BYTE *v2; // r13
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  __int64 *v5; // rbx
  __int64 *v6; // r13
  __int64 v7; // rsi

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( a2 )
  {
    v2 = a2;
    if ( *a2 != 5 )
      v2 = 0;
    if ( sub_B8D4C0(v2) )
    {
      sub_B8DE30(a1, (__int64)v2);
    }
    else
    {
      v3 = *(v2 - 16);
      if ( (v3 & 2) != 0 )
      {
        v5 = (__int64 *)*((_QWORD *)v2 - 4);
        v4 = *((unsigned int *)v2 - 6);
      }
      else
      {
        v4 = (*((_WORD *)v2 - 8) >> 6) & 0xF;
        v5 = (__int64 *)&v2[-8 * ((v3 >> 2) & 0xF) - 16];
      }
      v6 = &v5[v4];
      while ( v6 != v5 )
      {
        v7 = *v5++;
        sub_B8DE30(a1, v7);
      }
    }
  }
}
