// Function: sub_E321A0
// Address: 0xe321a0
//
void __fastcall sub_E321A0(__int64 a1)
{
  unsigned __int64 v1; // rdx
  unsigned __int64 v2; // rax
  __int64 v3; // rsi
  unsigned __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 v6; // rax

  if ( !*(_BYTE *)(a1 + 49) )
  {
    v1 = *(_QWORD *)(a1 + 40);
    v2 = *(_QWORD *)(a1 + 24);
    if ( v1 < v2 )
    {
      v3 = *(_QWORD *)(a1 + 32);
      if ( *(_BYTE *)(v3 + v1) == 71 )
      {
        *(_QWORD *)(a1 + 40) = v1 + 1;
        if ( v2 > v1 + 1 && *(_BYTE *)(v3 + v1 + 1) == 95 )
        {
          v4 = 1;
          *(_QWORD *)(a1 + 40) = v1 + 2;
        }
        else
        {
          v6 = sub_E31BC0(a1);
          if ( *(_BYTE *)(a1 + 49) )
            return;
          if ( v6 == -1 )
            goto LABEL_14;
          v4 = v6 + 1;
          v2 = *(_QWORD *)(a1 + 24);
        }
        if ( v2 - *(_QWORD *)(a1 + 16) > v4 )
        {
          v5 = 0;
          sub_E31C60(a1, 4u, "for<");
          ++*(_QWORD *)(a1 + 16);
          while ( 1 )
          {
            ++v5;
            sub_E31E00(a1, 1);
            if ( v5 == v4 )
              break;
            ++*(_QWORD *)(a1 + 16);
            sub_E31C60(a1, 2u, ", ");
          }
          sub_E31C60(a1, 2u, "> ");
          return;
        }
LABEL_14:
        *(_BYTE *)(a1 + 49) = 1;
      }
    }
  }
}
