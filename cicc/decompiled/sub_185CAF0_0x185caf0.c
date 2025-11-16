// Function: sub_185CAF0
// Address: 0x185caf0
//
__int64 __fastcall sub_185CAF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  _QWORD *v6; // rbx
  char v7; // al
  _QWORD *v9; // rax
  char v10; // dl
  _QWORD *v11; // rax
  char v12; // dl
  _QWORD *v13; // rsi
  unsigned int v14; // edi
  _QWORD *v15; // rcx
  _QWORD *v16; // rsi
  unsigned int v17; // edi
  _QWORD *v18; // rcx

  if ( a1 )
  {
    v3 = a1;
    while ( 1 )
    {
      v6 = sub_1648700(v3);
      v7 = *((_BYTE *)v6 + 16);
      if ( v7 == 75 )
      {
        if ( *(_BYTE *)(*(v6 - 3) + 16LL) != 15 )
          return 0;
        goto LABEL_9;
      }
      if ( v7 == 56 )
      {
        if ( (*((_DWORD *)v6 + 5) & 0xFFFFFFFu) <= 2 )
          return 0;
        goto LABEL_9;
      }
      if ( v7 != 77 )
        return 0;
      v9 = *(_QWORD **)(a3 + 8);
      if ( *(_QWORD **)(a3 + 16) != v9 )
        goto LABEL_13;
      v13 = &v9[*(unsigned int *)(a3 + 28)];
      v14 = *(_DWORD *)(a3 + 28);
      if ( v9 != v13 )
      {
        v15 = 0;
        while ( v6 != (_QWORD *)*v9 )
        {
          if ( *v9 == -2 )
            v15 = v9;
          if ( v13 == ++v9 )
          {
            if ( !v15 )
              goto LABEL_36;
            *v15 = v6;
            --*(_DWORD *)(a3 + 32);
            ++*(_QWORD *)a3;
            goto LABEL_14;
          }
        }
        return 0;
      }
LABEL_36:
      if ( v14 < *(_DWORD *)(a3 + 24) )
      {
        *(_DWORD *)(a3 + 28) = v14 + 1;
        *v13 = v6;
        ++*(_QWORD *)a3;
      }
      else
      {
LABEL_13:
        sub_16CCBA0(a3, (__int64)v6);
        if ( !v10 )
          return 0;
      }
LABEL_14:
      v11 = *(_QWORD **)(a2 + 8);
      if ( *(_QWORD **)(a2 + 16) == v11 )
      {
        v16 = &v11[*(unsigned int *)(a2 + 28)];
        v17 = *(_DWORD *)(a2 + 28);
        if ( v11 != v16 )
        {
          v18 = 0;
          while ( v6 != (_QWORD *)*v11 )
          {
            if ( *v11 == -2 )
              v18 = v11;
            if ( v16 == ++v11 )
            {
              if ( !v18 )
                goto LABEL_38;
              *v18 = v6;
              --*(_DWORD *)(a2 + 32);
              ++*(_QWORD *)a2;
              goto LABEL_16;
            }
          }
          goto LABEL_9;
        }
LABEL_38:
        if ( v17 < *(_DWORD *)(a2 + 24) )
        {
          *(_DWORD *)(a2 + 28) = v17 + 1;
          *v16 = v6;
          ++*(_QWORD *)a2;
LABEL_16:
          if ( !(unsigned __int8)sub_185CAF0(v6[1], a2, a3) )
            return 0;
          goto LABEL_9;
        }
      }
      sub_16CCBA0(a2, (__int64)v6);
      if ( v12 )
        goto LABEL_16;
LABEL_9:
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return 1;
    }
  }
  return 1;
}
