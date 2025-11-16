// Function: sub_2D5C910
// Address: 0x2d5c910
//
__int64 __fastcall sub_2D5C910(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rax
  __int64 v4; // r12
  __int64 i; // r14
  __int64 v6; // rbx
  __int64 j; // r15
  char v8; // al
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // [rsp+20h] [rbp-40h]

  v2 = *(_DWORD **)(a1 + 8);
  if ( v2 && *v2 == 2 )
  {
    v4 = *(_QWORD *)(a2 + 80);
    for ( i = 0; a2 + 72 != v4; v4 = *(_QWORD *)(v4 + 8) )
    {
      if ( !v4 )
        BUG();
      v6 = *(_QWORD *)(v4 + 32);
      for ( j = v4 + 24; j != v6; i += v9 )
      {
        while ( 1 )
        {
          if ( !v6 )
            BUG();
          v8 = *(_BYTE *)(v6 - 24);
          if ( v8 == 85 || v8 == 34 )
          {
            v9 = sub_D84370(a1, v6 - 24, 0, 0);
            if ( v10 )
              break;
          }
          v6 = *(_QWORD *)(v6 + 8);
          if ( j == v6 )
            goto LABEL_14;
        }
        v6 = *(_QWORD *)(v6 + 8);
      }
LABEL_14:
      ;
    }
    return i;
  }
  return v11;
}
