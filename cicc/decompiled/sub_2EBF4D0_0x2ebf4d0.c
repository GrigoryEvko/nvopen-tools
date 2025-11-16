// Function: sub_2EBF4D0
// Address: 0x2ebf4d0
//
void __fastcall sub_2EBF4D0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int16 v4; // dx
  char *v5; // rax
  char *v6; // r15
  char *v7; // r12
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  signed __int64 v11; // rdx
  __int64 v12; // rcx

  if ( a2 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  if ( v2 )
  {
    if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
    {
      while ( 1 )
      {
LABEL_5:
        v3 = *(_QWORD *)(v2 + 16);
        while ( 1 )
        {
          v2 = *(_QWORD *)(v2 + 32);
          if ( !v2 )
            break;
          if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 && v3 != *(_QWORD *)(v2 + 16) )
          {
            v4 = *(_WORD *)(v3 + 68);
            if ( v4 == 14 )
              goto LABEL_17;
            goto LABEL_9;
          }
        }
        v4 = *(_WORD *)(v3 + 68);
        if ( v4 == 14 )
        {
LABEL_17:
          v5 = *(char **)(v3 + 32);
          v6 = v5 + 40;
          v7 = v5;
          goto LABEL_18;
        }
LABEL_9:
        if ( v4 != 15 )
          goto LABEL_10;
        v8 = *(_QWORD *)(v3 + 32);
        v9 = 40LL * (*(_DWORD *)(v3 + 40) & 0xFFFFFF);
        v6 = (char *)(v8 + v9);
        v7 = (char *)(v8 + 80);
        v10 = (v9 - 80) >> 3;
        v5 = v7;
        v11 = 0xCCCCCCCCCCCCCCCDLL * v10;
        v12 = v11 >> 2;
        if ( v11 >> 2 > 0 )
        {
          while ( *v5 || *((_DWORD *)v5 + 2) != a2 )
          {
            if ( !v5[40] && *((_DWORD *)v5 + 12) == a2 )
            {
              if ( v5 + 40 == v6 )
                goto LABEL_10;
              goto LABEL_21;
            }
            if ( !v5[80] && *((_DWORD *)v5 + 22) == a2 )
            {
              if ( v5 + 80 == v6 )
                goto LABEL_10;
              goto LABEL_21;
            }
            if ( !v5[120] && *((_DWORD *)v5 + 32) == a2 )
            {
              if ( v5 + 120 == v6 )
                goto LABEL_10;
              goto LABEL_21;
            }
            v5 += 160;
            if ( !--v12 )
            {
              v11 = 0xCCCCCCCCCCCCCCCDLL * ((v6 - v5) >> 3);
              goto LABEL_37;
            }
          }
          goto LABEL_20;
        }
LABEL_37:
        if ( v11 != 2 )
        {
          if ( v11 != 3 )
          {
            if ( v11 != 1 )
              goto LABEL_10;
            goto LABEL_18;
          }
          if ( !*v5 && *((_DWORD *)v5 + 2) == a2 )
            goto LABEL_20;
          v5 += 40;
        }
        if ( !*v5 && *((_DWORD *)v5 + 2) == a2 )
          goto LABEL_20;
        v5 += 40;
LABEL_18:
        if ( *v5 || *((_DWORD *)v5 + 2) != a2 )
          goto LABEL_10;
LABEL_20:
        if ( v5 != v6 )
        {
LABEL_21:
          while ( v6 != v7 )
          {
            if ( !*v7 )
            {
              sub_2EAB0C0((__int64)v7, 0);
              *(_DWORD *)v7 &= 0xFFF000FF;
            }
            v7 += 40;
          }
        }
LABEL_10:
        if ( !v2 )
          return;
      }
    }
    while ( 1 )
    {
      v2 = *(_QWORD *)(v2 + 32);
      if ( !v2 )
        break;
      if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
        goto LABEL_5;
    }
  }
}
