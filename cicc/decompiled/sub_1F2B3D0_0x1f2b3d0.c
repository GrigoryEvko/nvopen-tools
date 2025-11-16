// Function: sub_1F2B3D0
// Address: 0x1f2b3d0
//
__int64 __fastcall sub_1F2B3D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 i; // r14
  _QWORD *v6; // rsi
  unsigned __int8 v7; // al
  __int64 v8; // rax
  _QWORD *v10; // rdi
  unsigned int v11; // r8d
  _QWORD *v12; // rcx
  _QWORD *v13; // rax
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rax
  int v17; // eax

  v2 = a1 + 296;
  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v6 = sub_1648700(i);
      v7 = *((_BYTE *)v6 + 16);
      if ( v7 <= 0x17u )
        goto LABEL_10;
      if ( v7 == 55 )
      {
        v8 = *(v6 - 6);
        if ( v8 && a2 == v8 )
          return 1;
        goto LABEL_10;
      }
      if ( v7 != 69 )
        break;
      v14 = *(v6 - 3);
      if ( a2 == v14 && v14 )
        return 1;
LABEL_10:
      i = *(_QWORD *)(i + 8);
      if ( !i )
        return 0;
    }
    if ( v7 == 78 )
    {
      v16 = *(v6 - 3);
      if ( *(_BYTE *)(v16 + 16) )
        return 1;
      if ( (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
        return 1;
      v17 = *(_DWORD *)(v16 + 36);
      if ( (unsigned int)(v17 - 35) > 3 && (unsigned int)(v17 - 116) > 1 )
        return 1;
      goto LABEL_10;
    }
    if ( v7 == 29 )
      return 1;
    if ( v7 != 79 )
    {
      if ( v7 == 77 )
      {
        v13 = *(_QWORD **)(a1 + 304);
        if ( *(_QWORD **)(a1 + 312) != v13 )
          goto LABEL_31;
        v10 = &v13[*(unsigned int *)(a1 + 324)];
        v11 = *(_DWORD *)(a1 + 324);
        if ( v13 != v10 )
        {
          v12 = 0;
          while ( v6 != (_QWORD *)*v13 )
          {
            if ( *v13 == -2 )
              v12 = v13;
            if ( v10 == ++v13 )
            {
              if ( !v12 )
                goto LABEL_38;
              *v12 = v6;
              --*(_DWORD *)(a1 + 328);
              ++*(_QWORD *)(a1 + 296);
              goto LABEL_24;
            }
          }
          goto LABEL_10;
        }
LABEL_38:
        if ( v11 < *(_DWORD *)(a1 + 320) )
        {
          *(_DWORD *)(a1 + 324) = v11 + 1;
          *v10 = v6;
          ++*(_QWORD *)(a1 + 296);
        }
        else
        {
LABEL_31:
          sub_16CCBA0(v2, (__int64)v6);
          if ( !v15 )
            goto LABEL_10;
        }
      }
      else if ( v7 != 56 && v7 != 71 )
      {
        goto LABEL_10;
      }
    }
LABEL_24:
    if ( (unsigned __int8)sub_1F2B3D0(a1, v6) )
      return 1;
  }
  return 0;
}
