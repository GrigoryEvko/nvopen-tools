// Function: sub_8627E0
// Address: 0x8627e0
//
void __fastcall sub_8627E0(__int64 a1)
{
  __int64 i; // r12
  __int64 v3; // rax
  __int64 j; // rbx
  int v5; // edi
  _QWORD *k; // rbx

  for ( i = *(_QWORD *)(a1 + 104); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
    {
      v3 = *(_QWORD *)(*(_QWORD *)(i + 168) + 152LL);
      if ( v3 )
      {
        if ( (*(_BYTE *)(v3 + 29) & 0x20) == 0 )
        {
          for ( j = *(_QWORD *)(v3 + 144); j; j = *(_QWORD *)(j + 112) )
          {
            while ( 1 )
            {
              v5 = *(_DWORD *)(j + 160);
              if ( v5 )
                break;
              j = *(_QWORD *)(j + 112);
              if ( !j )
                goto LABEL_10;
            }
            sub_862730(v5, 0);
          }
        }
      }
    }
LABEL_10:
    ;
  }
  for ( k = *(_QWORD **)(a1 + 160); k; k = (_QWORD *)*k )
    sub_8627E0(k);
}
