// Function: sub_AD7630
// Address: 0xad7630
//
_BYTE *__fastcall sub_AD7630(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 *v4; // rax
  __int64 *v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rax
  char *v16; // rax
  char *v17; // rcx
  __int64 v18; // rdx
  char *v19; // rdx
  signed __int64 v20; // rdx
  int v21; // [rsp+Ch] [rbp-34h]

  v3 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 14 )
    return (_BYTE *)sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 24LL), a2);
  switch ( v3 )
  {
    case 17:
      v4 = (__int64 *)sub_BD5C60(a1, a2, a3);
      return (_BYTE *)sub_ACCFD0(v4, a1 + 24);
    case 18:
      v6 = (__int64 *)sub_BD5C60(a1, a2, a3);
      return (_BYTE *)sub_AC8EA0(v6, (__int64 *)(a1 + 24));
    case 16:
      return (_BYTE *)sub_AD7600(a1);
    case 11:
      return sub_AC3A60(a1, a2);
  }
  v7 = 0;
  if ( v3 == 5 && *(_WORD *)(a1 + 2) == 63 )
  {
    v8 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    if ( (unsigned int)**(unsigned __int8 **)(a1 + 32 * (1 - v8)) - 12 <= 1 )
    {
      v9 = *(_QWORD *)(a1 - 32 * v8);
      if ( *(_BYTE *)v9 == 5
        && *(_WORD *)(v9 + 2) == 62
        && (unsigned int)**(unsigned __int8 **)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)) - 12 <= 1 )
      {
        v10 = sub_AC35F0(a1);
        v12 = v11;
        v13 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
        v14 = *(_QWORD *)(v9 + 32 * (2 - v13));
        if ( *(_BYTE *)v14 != 17 )
          return 0;
        if ( *(_DWORD *)(v14 + 32) > 0x40u )
        {
          v21 = *(_DWORD *)(v14 + 32);
          if ( v21 - (unsigned int)sub_C444A0(v14 + 24) > 0x40 )
            return 0;
          v15 = **(_QWORD **)(v14 + 24);
        }
        else
        {
          v15 = *(_QWORD *)(v14 + 24);
        }
        if ( v15 )
          return 0;
        v16 = (char *)v10;
        v17 = (char *)(v10 + 4 * v12);
        v18 = (4 * v12) >> 4;
        if ( v18 > 0 )
        {
          v19 = (char *)(v10 + 16 * v18);
          while ( !*(_DWORD *)v16 )
          {
            if ( *((_DWORD *)v16 + 1) )
            {
              v16 += 4;
              break;
            }
            if ( *((_DWORD *)v16 + 2) )
            {
              v16 += 8;
              break;
            }
            if ( *((_DWORD *)v16 + 3) )
            {
              v16 += 12;
              break;
            }
            v16 += 16;
            if ( v19 == v16 )
              goto LABEL_34;
          }
LABEL_30:
          if ( v17 != v16 )
            return 0;
          return *(_BYTE **)(v9 + 32 * (1 - v13));
        }
LABEL_34:
        v20 = v17 - v16;
        if ( v17 - v16 != 8 )
        {
          if ( v20 != 12 )
          {
            if ( v20 != 4 )
              return *(_BYTE **)(v9 + 32 * (1 - v13));
            goto LABEL_37;
          }
          if ( *(_DWORD *)v16 )
            goto LABEL_30;
          v16 += 4;
        }
        if ( *(_DWORD *)v16 )
          goto LABEL_30;
        v16 += 4;
LABEL_37:
        if ( !*(_DWORD *)v16 )
          return *(_BYTE **)(v9 + 32 * (1 - v13));
        goto LABEL_30;
      }
    }
  }
  return (_BYTE *)v7;
}
