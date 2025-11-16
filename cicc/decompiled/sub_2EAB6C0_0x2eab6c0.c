// Function: sub_2EAB6C0
// Address: 0x2eab6c0
//
__int64 __fastcall sub_2EAB6C0(__int64 a1, char *a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rdx
  __int64 v6; // rcx
  const void *v7; // r13
  const void *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  size_t v12; // rdx
  __int64 v13; // rdx
  size_t v14; // rdx

  v2 = *a2;
  if ( *a2 != *(_BYTE *)a1 )
    return 0;
  if ( v2 )
  {
    v3 = 0;
    v5 = (*(_DWORD *)a1 >> 8) & 0xFFF;
    v6 = (*(_DWORD *)a2 >> 8) & 0xFFF;
    if ( (_DWORD)v6 == (_DWORD)v5 )
    {
      switch ( v2 )
      {
        case 1:
        case 2:
        case 3:
        case 4:
          LOBYTE(v3) = *(_QWORD *)(a1 + 24) == *((_QWORD *)a2 + 3);
          return v3;
        case 5:
        case 8:
          LOBYTE(v3) = *(_DWORD *)(a1 + 24) == *((_DWORD *)a2 + 6);
          return v3;
        case 6:
        case 7:
          v3 = 0;
          if ( *(_DWORD *)(a1 + 24) == *((_DWORD *)a2 + 6) )
            goto LABEL_15;
          return v3;
        case 9:
          if ( strcmp(*(const char **)(a1 + 24), *((const char **)a2 + 3)) )
            goto LABEL_23;
          goto LABEL_15;
        case 10:
          if ( *(_QWORD *)(a1 + 24) == *((_QWORD *)a2 + 3) )
            goto LABEL_15;
          goto LABEL_23;
        case 11:
          if ( *((_QWORD *)a2 + 3) != *(_QWORD *)(a1 + 24) )
            goto LABEL_23;
LABEL_15:
          LOBYTE(v3) = (*((unsigned int *)a2 + 2) | (unsigned __int64)((__int64)*((int *)a2 + 8) << 32)) == (*(unsigned int *)(a1 + 8) | (unsigned __int64)((__int64)*(int *)(a1 + 32) << 32));
          break;
        case 12:
        case 13:
          v7 = *(const void **)(a1 + 24);
          v8 = (const void *)*((_QWORD *)a2 + 3);
          if ( v8 == v7 )
            goto LABEL_33;
          v9 = *(_QWORD *)(a1 + 16);
          if ( v9 && (v10 = *(_QWORD *)(v9 + 24)) != 0 )
          {
            v11 = *(_QWORD *)(v10 + 32);
            v3 = 0;
            if ( v11 )
            {
              v12 = 4LL
                  * ((unsigned int)(*(_DWORD *)((*(__int64 (__fastcall **)(_QWORD, char *, __int64, __int64, _QWORD))(**(_QWORD **)(v11 + 16) + 200LL))(
                                                  *(_QWORD *)(v11 + 16),
                                                  a2,
                                                  v5,
                                                  v6,
                                                  0)
                                              + 16)
                                  + 31) >> 5);
              if ( v12 )
                LOBYTE(v3) = memcmp(v7, v8, v12) == 0;
              else
LABEL_33:
                v3 = 1;
            }
          }
          else
          {
LABEL_23:
            v3 = 0;
          }
          break;
        case 14:
        case 15:
          LOBYTE(v3) = *((_QWORD *)a2 + 3) == *(_QWORD *)(a1 + 24);
          return v3;
        case 16:
        case 17:
        case 18:
          LOBYTE(v3) = *((_DWORD *)a2 + 6) == *(_DWORD *)(a1 + 24);
          return v3;
        case 19:
          v13 = *(_QWORD *)(a1 + 32);
          v3 = 0;
          if ( *((_QWORD *)a2 + 4) == v13 )
          {
            v14 = 4 * v13;
            v3 = 1;
            if ( v14 )
              LOBYTE(v3) = memcmp(*(const void **)(a1 + 24), *((const void **)a2 + 3), v14) == 0;
          }
          return v3;
        case 20:
          v3 = 0;
          if ( *((_DWORD *)a2 + 6) == *(_DWORD *)(a1 + 24) )
            LOBYTE(v3) = *((_DWORD *)a2 + 7) == *(_DWORD *)(a1 + 28);
          return v3;
        default:
          BUG();
      }
    }
  }
  else
  {
    v3 = 0;
    if ( *(_DWORD *)(a1 + 8) == *((_DWORD *)a2 + 2) && ((*(_BYTE *)(a1 + 3) & 0x10) != 0) == ((a2[3] & 0x10) != 0) )
      LOBYTE(v3) = ((*(_DWORD *)a1 >> 8) & 0xFFF) == ((*(_DWORD *)a2 >> 8) & 0xFFF);
  }
  return v3;
}
