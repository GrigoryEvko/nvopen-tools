// Function: sub_2CE3810
// Address: 0x2ce3810
//
void __fastcall sub_2CE3810(_QWORD *a1, __int64 a2)
{
  __int64 i; // r12
  __int64 v3; // r15
  char v4; // al
  __int64 v5; // rax
  int v6; // edx
  __int64 *v7; // rbx
  __int64 v8; // r13
  char v9; // al
  __int64 v10; // rax
  int v11; // edx
  __int64 *v12; // rbx
  __int64 v13; // r13
  char v14; // al

  for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v3 = *(_QWORD *)(i + 24);
    v4 = *(_BYTE *)v3;
    if ( *(_BYTE *)v3 <= 0x1Cu )
      BUG();
    if ( v4 == 61 )
    {
      v5 = *(_QWORD *)(v3 + 8);
      if ( *(_BYTE *)(v5 + 8) == 15 )
      {
        v6 = *(_DWORD *)(v5 + 12);
        if ( v6 )
        {
          v7 = *(__int64 **)(v5 + 16);
          v8 = (__int64)&v7[(unsigned int)(v6 - 1) + 1];
          while ( 1 )
          {
            v9 = *(_BYTE *)(*v7 + 8);
            if ( v9 == 14 || v9 == 15 && (unsigned __int8)sub_2CDFA60(*v7) )
              break;
            if ( (__int64 *)v8 == ++v7 )
              goto LABEL_12;
          }
          sub_2CE3690(a1, v3);
        }
      }
    }
    else if ( v4 == 62 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(v3 - 64) + 8LL);
      if ( *(_BYTE *)(v10 + 8) == 15 )
      {
        v11 = *(_DWORD *)(v10 + 12);
        if ( v11 )
        {
          v12 = *(__int64 **)(v10 + 16);
          v13 = (__int64)&v12[(unsigned int)(v11 - 1) + 1];
          while ( 1 )
          {
            v14 = *(_BYTE *)(*v12 + 8);
            if ( v14 == 14 || v14 == 15 && (unsigned __int8)sub_2CDFA60(*v12) )
              break;
            if ( (__int64 *)v13 == ++v12 )
              goto LABEL_12;
          }
          sub_2CE2350((__int64)a1, v3);
        }
      }
    }
LABEL_12:
    ;
  }
}
