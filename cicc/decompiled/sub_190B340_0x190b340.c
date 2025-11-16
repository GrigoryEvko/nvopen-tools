// Function: sub_190B340
// Address: 0x190b340
//
__int64 __fastcall sub_190B340(__int64 a1, __int64 a2)
{
  int v2; // r13d
  unsigned int v3; // eax
  __int64 v5; // rbx
  unsigned int v6; // r11d
  unsigned int v7; // r15d
  __int64 v8; // rdx
  __int64 *v9; // rdx
  __int64 v10; // r10
  char v11; // si
  __int64 v12; // rcx
  int v13; // r8d
  unsigned int v14; // r9d
  __int64 *v15; // rax
  __int64 v16; // r14
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r8
  int v26; // eax
  int v27; // [rsp+0h] [rbp-34h]

  v2 = *(_DWORD *)(a2 + 20);
  v3 = v2 & 0xFFFFFFF;
  if ( (v2 & 0xFFFFFFF) != 0 )
  {
    v5 = 0;
    v6 = 0;
    v7 = 0;
    while ( 1 )
    {
      v8 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * v3;
      v9 = (__int64 *)(v5 + v8);
      v10 = *v9;
      v11 = *(_BYTE *)(a1 + 520) & 1;
      if ( v11 )
      {
        v12 = a1 + 528;
        v13 = 3;
      }
      else
      {
        v23 = *(unsigned int *)(a1 + 536);
        v12 = *(_QWORD *)(a1 + 528);
        if ( !(_DWORD)v23 )
          goto LABEL_25;
        v13 = v23 - 1;
      }
      v14 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v10 != *v15 )
        break;
LABEL_7:
      v17 = 64;
      if ( !v11 )
        v17 = 16LL * *(unsigned int *)(a1 + 536);
      if ( v15 != (__int64 *)(v17 + v12) )
      {
        v18 = *(_QWORD *)(a1 + 592) + 16LL * *((unsigned int *)v15 + 2);
        if ( v18 != *(_QWORD *)(a1 + 592) + 16LL * *(unsigned int *)(a1 + 600) )
        {
          v19 = *(_QWORD *)(v18 + 8);
          if ( v10 )
          {
            v20 = v9[1];
            v21 = v9[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v21 = v20;
            if ( v20 )
              *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
          }
          *v9 = v19;
          if ( v19 )
          {
            v22 = *(_QWORD *)(v19 + 8);
            v9[1] = v22;
            if ( v22 )
              *(_QWORD *)(v22 + 16) = (unsigned __int64)(v9 + 1) | *(_QWORD *)(v22 + 16) & 3LL;
            v7 = 1;
            v9[2] = (v19 + 8) | v9[2] & 3;
            *(_QWORD *)(v19 + 8) = v9;
            v2 = *(_DWORD *)(a2 + 20);
          }
          else
          {
            v2 = *(_DWORD *)(a2 + 20);
            v7 = 1;
          }
        }
      }
      ++v6;
      v5 += 24;
      v3 = v2 & 0xFFFFFFF;
      if ( (v2 & 0xFFFFFFFu) <= v6 )
        return v7;
    }
    v26 = 1;
    while ( v16 != -8 )
    {
      v14 = v13 & (v26 + v14);
      v27 = v26 + 1;
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
        goto LABEL_7;
      v26 = v27;
    }
    if ( v11 )
    {
      v24 = 64;
    }
    else
    {
      v23 = *(unsigned int *)(a1 + 536);
LABEL_25:
      v24 = 16 * v23;
    }
    v15 = (__int64 *)(v12 + v24);
    goto LABEL_7;
  }
  return 0;
}
