// Function: sub_1E31610
// Address: 0x1e31610
//
__int64 __fastcall sub_1E31610(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v4; // rdx
  __int64 v5; // rcx
  const void *v6; // r13
  const void *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  size_t v13; // rdx

  if ( *(_BYTE *)a1 != *(_BYTE *)a2 )
    return 0;
  if ( *(_BYTE *)a1 )
  {
    v2 = 0;
    v4 = (*(_DWORD *)a1 >> 8) & 0xFFF;
    v5 = (*(_DWORD *)a2 >> 8) & 0xFFF;
    if ( (_DWORD)v5 == (_DWORD)v4 )
    {
      switch ( *(_BYTE *)a1 )
      {
        case 1:
        case 2:
          LOBYTE(v2) = *(_QWORD *)(a1 + 24) == *(_QWORD *)(a2 + 24);
          return v2;
        case 3:
        case 4:
        case 0xE:
        case 0xF:
          LOBYTE(v2) = *(_QWORD *)(a2 + 24) == *(_QWORD *)(a1 + 24);
          return v2;
        case 5:
        case 8:
          LOBYTE(v2) = *(_DWORD *)(a1 + 24) == *(_DWORD *)(a2 + 24);
          return v2;
        case 6:
        case 7:
          v2 = 0;
          if ( *(_DWORD *)(a2 + 24) == *(_DWORD *)(a1 + 24) )
            goto LABEL_23;
          return v2;
        case 9:
          if ( !strcmp(*(const char **)(a1 + 24), *(const char **)(a2 + 24)) )
            goto LABEL_23;
          goto LABEL_25;
        case 0xA:
          if ( *(_QWORD *)(a1 + 24) != *(_QWORD *)(a2 + 24) )
            goto LABEL_25;
          goto LABEL_23;
        case 0xB:
          if ( *(_QWORD *)(a2 + 24) != *(_QWORD *)(a1 + 24) )
            goto LABEL_25;
LABEL_23:
          LOBYTE(v2) = (*(unsigned int *)(a2 + 8) | (unsigned __int64)((__int64)*(int *)(a2 + 32) << 32)) == (*(unsigned int *)(a1 + 8) | (unsigned __int64)((__int64)*(int *)(a1 + 32) << 32));
          break;
        case 0xC:
        case 0xD:
          v6 = *(const void **)(a1 + 24);
          v7 = *(const void **)(a2 + 24);
          if ( v7 == v6 )
            goto LABEL_30;
          v8 = *(_QWORD *)(a1 + 16);
          if ( v8 && (v9 = *(_QWORD *)(v8 + 24)) != 0 && (v10 = *(_QWORD *)(v9 + 56)) != 0 )
          {
            v11 = *(_QWORD *)(v10 + 16);
            v12 = *(__int64 (**)())(*(_QWORD *)v11 + 112LL);
            if ( v12 == sub_1D00B10 )
              BUG();
            v13 = 4LL
                * ((unsigned int)(*(_DWORD *)(((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v12)(
                                                v11,
                                                a2,
                                                v4,
                                                v5,
                                                0)
                                            + 16)
                                + 31) >> 5);
            if ( v13 )
              LOBYTE(v2) = memcmp(v6, v7, v13) == 0;
            else
LABEL_30:
              v2 = 1;
          }
          else
          {
LABEL_25:
            v2 = 0;
          }
          break;
        case 0x10:
        case 0x11:
        case 0x12:
          LOBYTE(v2) = *(_DWORD *)(a2 + 24) == *(_DWORD *)(a1 + 24);
          return v2;
      }
    }
  }
  else
  {
    v2 = 0;
    if ( *(_DWORD *)(a1 + 8) == *(_DWORD *)(a2 + 8)
      && ((*(_BYTE *)(a2 + 3) & 0x10) != 0) == ((*(_BYTE *)(a1 + 3) & 0x10) != 0) )
    {
      LOBYTE(v2) = ((*(_DWORD *)a2 >> 8) & 0xFFF) == ((*(_DWORD *)a1 >> 8) & 0xFFF);
    }
  }
  return v2;
}
