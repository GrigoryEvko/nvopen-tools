// Function: sub_2A457F0
// Address: 0x2a457f0
//
__int64 __fastcall sub_2A457F0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v4; // rbx
  __int64 v5; // rdx
  char v6; // r13
  __int64 v7; // r14
  __int64 v9; // rax
  int v10; // eax
  int v11; // edi

  v2 = *(_DWORD *)(a2 + 24);
  if ( v2 > 1 )
  {
    if ( v2 != 2 )
      BUG();
    if ( *(_QWORD *)(a2 + 48) != *(_QWORD *)(a2 + 40) )
      goto LABEL_7;
    v9 = *(_QWORD *)(a2 + 72);
  }
  else
  {
    v4 = *(_QWORD *)(a2 + 48);
    v5 = *(_QWORD *)(a2 + 40);
    if ( v2 )
    {
      v6 = 1;
      if ( v4 != v5 )
      {
LABEL_4:
        if ( (unsigned __int8)(*(_BYTE *)v4 - 82) <= 1u )
        {
          v7 = *(_QWORD *)(v4 - 32);
          if ( *(_QWORD *)(v4 - 64) == v5 )
          {
            v11 = *(_WORD *)(v4 + 2) & 0x3F;
          }
          else
          {
            if ( v7 != v5 )
              goto LABEL_7;
            v10 = sub_B52F50(*(_WORD *)(v4 + 2) & 0x3F);
            v7 = *(_QWORD *)(v4 - 64);
            v11 = v10;
          }
          if ( !v6 )
            v11 = sub_B52870(v11);
          *(_DWORD *)a1 = v11;
          *(_QWORD *)(a1 + 8) = v7;
          *(_BYTE *)(a1 + 16) = 1;
          return a1;
        }
LABEL_7:
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
    }
    else
    {
      v6 = *(_BYTE *)(a2 + 72);
      if ( v4 != v5 )
        goto LABEL_4;
      if ( !v6 )
      {
        v9 = sub_AD6450(*(_QWORD *)(v4 + 8));
        goto LABEL_12;
      }
    }
    v9 = sub_AD6400(*(_QWORD *)(v5 + 8));
  }
LABEL_12:
  *(_QWORD *)(a1 + 8) = v9;
  *(_DWORD *)a1 = 32;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
