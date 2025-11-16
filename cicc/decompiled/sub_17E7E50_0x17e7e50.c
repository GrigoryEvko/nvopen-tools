// Function: sub_17E7E50
// Address: 0x17e7e50
//
void __fastcall sub_17E7E50(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r14
  __int64 v4; // rbx
  __int64 v5; // r15
  _QWORD *v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r10
  int v14; // eax
  int v15; // r9d

  v1 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v2 = *(_QWORD *)a1 + 72LL;
  if ( v1 != v2 )
  {
    while ( 1 )
    {
      v4 = v1 - 24;
      if ( !v1 )
        v4 = 0;
      if ( (unsigned __int8)sub_1368B40(*(__int64 **)(a1 + 16), v4) )
        goto LABEL_15;
      v5 = *(_QWORD *)(v4 + 8);
      if ( v5 )
      {
        while ( 1 )
        {
          v6 = sub_1648700(v5);
          if ( (unsigned __int8)(*((_BYTE *)v6 + 16) - 25) <= 9u )
            break;
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            goto LABEL_8;
        }
LABEL_11:
        if ( *(_BYTE *)(sub_157EBA0(v6[5]) + 16) == 28 )
        {
LABEL_15:
          v7 = sub_157EBA0(v4);
          v8 = *(unsigned int *)(a1 + 296);
          v9 = *(_QWORD *)(a1 + 280);
          v10 = v7;
          if ( !(_DWORD)v8 )
            goto LABEL_21;
          v11 = (v8 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v12 = (__int64 *)(v9 + 16LL * v11);
          v13 = *v12;
          if ( v4 != *v12 )
          {
            v14 = 1;
            while ( v13 != -8 )
            {
              v15 = v14 + 1;
              v11 = (v8 - 1) & (v14 + v11);
              v12 = (__int64 *)(v9 + 16LL * v11);
              v13 = *v12;
              if ( v4 == *v12 )
                goto LABEL_17;
              v14 = v15;
            }
LABEL_21:
            v12 = (__int64 *)(v9 + 16 * v8);
          }
LABEL_17:
          sub_17E7E10(*(__int64 **)(a1 + 8), v10, *(_QWORD *)(v12[1] + 16));
          v1 = *(_QWORD *)(v1 + 8);
          if ( v2 == v1 )
            return;
        }
        else
        {
          while ( 1 )
          {
            v5 = *(_QWORD *)(v5 + 8);
            if ( !v5 )
              break;
            v6 = sub_1648700(v5);
            if ( (unsigned __int8)(*((_BYTE *)v6 + 16) - 25) <= 9u )
              goto LABEL_11;
          }
          v1 = *(_QWORD *)(v1 + 8);
          if ( v2 == v1 )
            return;
        }
      }
      else
      {
LABEL_8:
        v1 = *(_QWORD *)(v1 + 8);
        if ( v2 == v1 )
          return;
      }
    }
  }
}
