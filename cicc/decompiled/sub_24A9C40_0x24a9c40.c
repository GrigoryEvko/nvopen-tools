// Function: sub_24A9C40
// Address: 0x24a9c40
//
void __fastcall sub_24A9C40(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 v14; // rdi
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r10
  int v18; // eax
  int v19; // r9d

  v1 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v2 = *(_QWORD *)a1 + 72LL;
  if ( v1 != v2 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
LABEL_2:
        v4 = v1 - 24;
        if ( !v1 )
          v4 = 0;
        if ( (unsigned __int8)sub_FDD330(*(__int64 **)(a1 + 16), v4) )
          break;
        v5 = *(_QWORD *)(v4 + 16);
        if ( v5 )
        {
          while ( 1 )
          {
            v6 = *(_QWORD *)(v5 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v6 - 30) <= 0xAu )
              break;
            v5 = *(_QWORD *)(v5 + 8);
            if ( !v5 )
            {
              v1 = *(_QWORD *)(v1 + 8);
              if ( v2 != v1 )
                goto LABEL_2;
              return;
            }
          }
LABEL_9:
          v7 = *(_QWORD *)(v6 + 40);
          v8 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v8 == v7 + 48 )
            goto LABEL_36;
          if ( !v8 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
LABEL_36:
            BUG();
          if ( *(_BYTE *)(v8 - 24) == 33 )
            break;
          while ( 1 )
          {
            v5 = *(_QWORD *)(v5 + 8);
            if ( !v5 )
              break;
            v6 = *(_QWORD *)(v5 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v6 - 30) <= 0xAu )
              goto LABEL_9;
          }
        }
        v1 = *(_QWORD *)(v1 + 8);
        if ( v2 == v1 )
          return;
      }
      v9 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v9 == v4 + 48 )
      {
        v12 = 0;
      }
      else
      {
        if ( !v9 )
          BUG();
        v10 = *(unsigned __int8 *)(v9 - 24);
        v11 = v9 - 24;
        if ( (unsigned int)(v10 - 30) >= 0xB )
          v11 = 0;
        v12 = v11;
      }
      v13 = *(_DWORD *)(a1 + 296);
      v14 = *(_QWORD *)(a1 + 280);
      if ( !v13 )
        goto LABEL_30;
      v15 = (v13 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v4 != *v16 )
        break;
LABEL_26:
      sub_24A9C00(*(__int64 **)(a1 + 8), v12, *(_QWORD *)(v16[1] + 16));
      v1 = *(_QWORD *)(v1 + 8);
      if ( v2 == v1 )
        return;
    }
    v18 = 1;
    while ( v17 != -4096 )
    {
      v19 = v18 + 1;
      v15 = (v13 - 1) & (v18 + v15);
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v4 == *v16 )
        goto LABEL_26;
      v18 = v19;
    }
LABEL_30:
    v16 = (__int64 *)(v14 + 16LL * v13);
    goto LABEL_26;
  }
}
