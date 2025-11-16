// Function: sub_1B43710
// Address: 0x1b43710
//
char __fastcall sub_1B43710(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rbx
  _QWORD *v5; // rax
  __int64 v6; // r15
  unsigned __int8 v7; // al
  __int64 v8; // r13
  __int64 v9; // rdi
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 **v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  int v18; // r12d
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // r12d
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-40h]

  v4 = (__int64 *)a2;
  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
  {
    while ( v4[1] && (sub_1593BB0(a1, a2, a3, a4) || *(_BYTE *)(a1 + 16) == 9) )
    {
      v5 = sub_1648700(v4[1]);
      v6 = v4[4];
      v23 = (__int64)v5;
      a4 = (__int64)v5;
      v7 = *((_BYTE *)v5 + 16);
      if ( v7 <= 0x17u )
      {
        if ( v6 )
        {
          v8 = 0;
          goto LABEL_6;
        }
        return 0;
      }
      v8 = a4 + 24;
      if ( v6 != a4 + 24 )
      {
LABEL_6:
        while ( v6 != v4[5] + 40 )
        {
          v9 = v6 - 24;
          if ( !v6 )
            v9 = 0;
          if ( (unsigned __int8)sub_15F3040(v9) || sub_15F3330(v9) )
            break;
          v6 = *(_QWORD *)(v6 + 8);
          if ( v8 == v6 )
          {
            v7 = *(_BYTE *)(v23 + 16);
            if ( v7 <= 0x17u )
              return 0;
            goto LABEL_14;
          }
        }
        return 0;
      }
LABEL_14:
      if ( v7 == 56 )
      {
        a2 = v23;
        v15 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        a3 = 4 * v15;
        v16 = *(__int64 **)(v23 - 24 * v15);
        if ( !v16 || v4 != v16 )
          return 0;
      }
      else if ( v7 != 71 )
      {
        if ( v7 == 54 )
        {
          if ( (*(_BYTE *)(v23 + 18) & 1) != 0 )
            return 0;
          v20 = **(_QWORD **)(v23 - 24);
          if ( *(_BYTE *)(v20 + 8) == 16 )
            v20 = **(_QWORD **)(v20 + 16);
          v21 = *(_DWORD *)(v20 + 8);
          v22 = sub_15F2060(v23);
          return sub_15E4690(v22, v21 >> 8) ^ 1;
        }
        if ( v7 == 55 )
        {
          if ( (*(_BYTE *)(v23 + 18) & 1) == 0 )
          {
            v17 = **(_QWORD **)(v23 - 24);
            if ( *(_BYTE *)(v17 + 8) == 16 )
              v17 = **(_QWORD **)(v17 + 16);
            v18 = *(_DWORD *)(v17 + 8) >> 8;
            v19 = sub_15F2060(v23);
            if ( !sub_15E4690(v19, v18) )
              return *(_QWORD *)(v23 - 24) != 0 && v4 == *(__int64 **)(v23 - 24);
          }
        }
        else
        {
          if ( v7 == 78 )
          {
            v11 = v23 | 4;
          }
          else
          {
            if ( v7 != 29 )
              return 0;
            v11 = v23 & 0xFFFFFFFFFFFFFFFBLL;
          }
          v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v13 = sub_15F2060(v11 & 0xFFFFFFFFFFFFFFF8LL);
            if ( !sub_15E4690(v13, 0) )
            {
              v14 = (__int64 **)(v12 - 72);
              if ( (v11 & 4) != 0 )
                v14 = (__int64 **)(v12 - 24);
              return *v14 == v4;
            }
          }
        }
        return 0;
      }
      if ( *(_BYTE *)(a1 + 16) > 0x10u )
        return 0;
      v4 = (__int64 *)v23;
    }
  }
  return 0;
}
