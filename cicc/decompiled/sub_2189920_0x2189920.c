// Function: sub_2189920
// Address: 0x2189920
//
char __fastcall sub_2189920(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  bool v5; // zf
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  _DWORD *v8; // rdx
  _DWORD *v9; // rdx
  __int64 v10; // rdx
  _DWORD *v11; // rdx
  _DWORD *v12; // rdx
  _DWORD *v13; // rdx
  __int64 v14; // rdx
  _DWORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx

  v5 = strcmp(a5, "ftz") == 0;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  LOBYTE(v7) = !v5;
  if ( v5 )
  {
    if ( (v6 & 0x10) != 0 )
    {
      v9 = *(_DWORD **)(a4 + 24);
      v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v9;
      if ( v7 <= 3 )
      {
        LOBYTE(v7) = sub_16E7EE0(a4, ".ftz", 4u);
      }
      else
      {
        *v9 = 2054448686;
        *(_QWORD *)(a4 + 24) += 4LL;
      }
    }
  }
  else
  {
    v5 = strcmp(a5, "sat") == 0;
    LOBYTE(v7) = !v5;
    if ( v5 )
    {
      if ( (v6 & 0x20) != 0 )
      {
        v8 = *(_DWORD **)(a4 + 24);
        v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".sat", 4u);
        }
        else
        {
          *v8 = 1952543534;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
      }
    }
    else
    {
      switch ( v6 & 0xF )
      {
        case 1LL:
          v11 = *(_DWORD **)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v11;
          if ( v7 <= 3 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rni", 4u);
          }
          else
          {
            *v11 = 1768845870;
            *(_QWORD *)(a4 + 24) += 4LL;
          }
          break;
        case 2LL:
          v13 = *(_DWORD **)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v13;
          if ( v7 <= 3 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rzi", 4u);
          }
          else
          {
            *v13 = 1769632302;
            *(_QWORD *)(a4 + 24) += 4LL;
          }
          break;
        case 3LL:
          v15 = *(_DWORD **)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v15;
          if ( v7 <= 3 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rmi", 4u);
          }
          else
          {
            *v15 = 1768780334;
            *(_QWORD *)(a4 + 24) += 4LL;
          }
          break;
        case 4LL:
          v12 = *(_DWORD **)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v12;
          if ( v7 <= 3 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rpi", 4u);
          }
          else
          {
            *v12 = 1768976942;
            *(_QWORD *)(a4 + 24) += 4LL;
          }
          break;
        case 5LL:
          v16 = *(_QWORD *)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - v16;
          if ( v7 <= 2 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rn", 3u);
          }
          else
          {
            *(_BYTE *)(v16 + 2) = 110;
            *(_WORD *)v16 = 29230;
            *(_QWORD *)(a4 + 24) += 3LL;
          }
          break;
        case 6LL:
          v14 = *(_QWORD *)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - v14;
          if ( v7 <= 2 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rz", 3u);
          }
          else
          {
            *(_BYTE *)(v14 + 2) = 122;
            *(_WORD *)v14 = 29230;
            *(_QWORD *)(a4 + 24) += 3LL;
          }
          break;
        case 7LL:
          v17 = *(_QWORD *)(a4 + 24);
          v7 = *(_QWORD *)(a4 + 16) - v17;
          if ( v7 <= 2 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rm", 3u);
          }
          else
          {
            *(_BYTE *)(v17 + 2) = 109;
            *(_WORD *)v17 = 29230;
            *(_QWORD *)(a4 + 24) += 3LL;
          }
          break;
        case 8LL:
          v10 = *(_QWORD *)(a4 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v10) <= 2 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".rp", 3u);
          }
          else
          {
            *(_BYTE *)(v10 + 2) = 112;
            *(_WORD *)v10 = 29230;
            *(_QWORD *)(a4 + 24) += 3LL;
            LOBYTE(v7) = 46;
          }
          break;
        default:
          return v7;
      }
    }
  }
  return v7;
}
