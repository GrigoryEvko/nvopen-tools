// Function: sub_5F6E90
// Address: 0x5f6e90
//
__int64 __fastcall sub_5F6E90(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r8
  unsigned __int64 v9; // rcx
  unsigned __int64 i; // rdx
  unsigned int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // rdi

  result = *(_QWORD *)(a2 + 24);
  if ( !result )
  {
    if ( (*(_BYTE *)(a2 + 32) & 1) == 0 && *(_QWORD *)(a2 + 16) )
    {
      v4 = *(_QWORD *)(a1 + 8);
      v5 = *(int *)(*(_QWORD *)(*(_QWORD *)(v4 + 168) + 152LL) + 240LL);
      if ( (int)v5 <= dword_4F04C64 && (_DWORD)v5 != -1 )
      {
        v6 = 776 * v5;
        if ( *(_BYTE *)(qword_4F04C68[0] + v6 + 4) == 6 && v4 == *(_QWORD *)(qword_4F04C68[0] + v6 + 208) )
        {
          v7 = *(int *)(qword_4F04C68[0] + v6 - 376);
          if ( (_DWORD)v7 != -1 )
          {
            while ( 2 )
            {
              v8 = 776 * v7;
              v9 = *(_QWORD *)(qword_4F04C68[0] + v8 + 216);
              if ( v9 && (*(_BYTE *)(v9 + 206) & 2) != 0 )
              {
                for ( i = v9 >> 3; ; LODWORD(i) = v11 + 1 )
                {
                  v11 = qword_4F04C10[1] & i;
                  v12 = (_QWORD *)(*qword_4F04C10 + 16LL * v11);
                  if ( v9 == *v12 )
                    break;
                  if ( !*v12 )
                    goto LABEL_16;
                }
                v13 = v12[1];
                if ( v13 )
                {
                  *(_QWORD *)(a2 + 16) = sub_5F6E90(v13, *(_QWORD *)(a2 + 16));
                  goto LABEL_18;
                }
LABEL_16:
                v7 = *(int *)(qword_4F04C68[0] + v8 - 376);
                if ( (_DWORD)v7 != -1 )
                  continue;
              }
              break;
            }
          }
        }
      }
      *(_QWORD *)(a2 + 16) = 0;
    }
LABEL_18:
    *(_BYTE *)(a2 + 33) &= ~2u;
    result = sub_5F6470(a1, (__int64 *)a2);
    *(_QWORD *)(a2 + 24) = result;
  }
  return result;
}
