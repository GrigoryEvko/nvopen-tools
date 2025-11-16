// Function: sub_27DDD70
// Address: 0x27ddd70
//
__int64 __fastcall sub_27DDD70(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 result; // rax
  int v5; // esi
  __int64 v6; // r10
  __int64 v7; // r9
  __int64 v8; // r11
  _QWORD *v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rbx
  unsigned __int64 v12; // rbx

  v3 = **(_QWORD **)(a2 - 8);
  result = 0;
  if ( *(_BYTE *)v3 == 84 && a3 == *(_QWORD *)(v3 + 40) )
  {
    v5 = *(_DWORD *)(v3 + 4) & 0x7FFFFFF;
    if ( v5 )
    {
      v6 = *(_QWORD *)(v3 - 8);
      v7 = 0;
      v8 = v6 + 32LL * *(unsigned int *)(v3 + 72);
      while ( 1 )
      {
        v9 = *(_QWORD **)(v6 + 32 * v7);
        if ( *(_BYTE *)v9 == 86 )
          break;
        if ( v5 == (_DWORD)++v7 )
          return 0;
      }
LABEL_8:
      v10 = *(_QWORD *)(v8 + 8 * v7);
      if ( v10 != v9[5] )
        goto LABEL_27;
      v11 = v9[2];
      if ( !v11 || *(_QWORD *)(v11 + 8) )
        goto LABEL_27;
      v12 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v12 == v10 + 48 )
        goto LABEL_24;
      if ( !v12 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 > 0xA )
LABEL_24:
        BUG();
      if ( *(_BYTE *)(v12 - 24) == 31 && (*(_DWORD *)(v12 - 20) & 0x7FFFFFF) == 1 )
      {
        sub_27DD6D0(a1, *(_QWORD *)(v8 + 8 * v7), a3, v9, v3, v7);
        return 1;
      }
      else
      {
LABEL_27:
        while ( v5 != (_DWORD)++v7 )
        {
          v9 = *(_QWORD **)(v6 + 32 * v7);
          if ( *(_BYTE *)v9 == 86 )
            goto LABEL_8;
        }
        return 0;
      }
    }
  }
  return result;
}
