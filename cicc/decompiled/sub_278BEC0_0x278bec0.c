// Function: sub_278BEC0
// Address: 0x278bec0
//
__int64 __fastcall sub_278BEC0(__int64 a1, __int64 a2)
{
  int v2; // r12d
  unsigned int v3; // eax
  __int64 v5; // r10
  unsigned int v6; // r14d
  __int64 v7; // rdx
  __int64 *v8; // rdx
  __int64 v9; // r9
  char v10; // si
  __int64 v11; // r11
  int v12; // r8d
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v23; // r8
  int v24; // eax
  int v25; // [rsp+0h] [rbp-2Ch]

  v2 = *(_DWORD *)(a2 + 4);
  v3 = v2 & 0x7FFFFFF;
  if ( (v2 & 0x7FFFFFF) != 0 )
  {
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      v7 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * v3;
      v8 = (__int64 *)(32 * v5 + v7);
      v9 = *v8;
      v10 = *(_BYTE *)(a1 + 496) & 1;
      if ( v10 )
      {
        v11 = a1 + 504;
        v12 = 3;
      }
      else
      {
        v21 = *(unsigned int *)(a1 + 512);
        v11 = *(_QWORD *)(a1 + 504);
        if ( !(_DWORD)v21 )
          goto LABEL_27;
        v12 = v21 - 1;
      }
      v13 = v12 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( v9 != *v14 )
        break;
LABEL_7:
      v16 = 64;
      if ( !v10 )
        v16 = 16LL * *(unsigned int *)(a1 + 512);
      if ( v14 != (__int64 *)(v11 + v16) )
      {
        v17 = *(_QWORD *)(a1 + 568) + 16LL * *((unsigned int *)v14 + 2);
        if ( v17 != *(_QWORD *)(a1 + 568) + 16LL * *(unsigned int *)(a1 + 576) )
        {
          v18 = *(_QWORD *)(v17 + 8);
          if ( v9 )
          {
            v19 = v8[1];
            *(_QWORD *)v8[2] = v19;
            if ( v19 )
              *(_QWORD *)(v19 + 16) = v8[2];
          }
          *v8 = v18;
          if ( v18 )
          {
            v20 = *(_QWORD *)(v18 + 16);
            v8[1] = v20;
            if ( v20 )
              *(_QWORD *)(v20 + 16) = v8 + 1;
            v8[2] = v18 + 16;
            v6 = 1;
            *(_QWORD *)(v18 + 16) = v8;
            v2 = *(_DWORD *)(a2 + 4);
          }
          else
          {
            v2 = *(_DWORD *)(a2 + 4);
            v6 = 1;
          }
        }
      }
      ++v5;
      v3 = v2 & 0x7FFFFFF;
      if ( (v2 & 0x7FFFFFFu) <= (unsigned int)v5 )
        return v6;
    }
    v24 = 1;
    while ( v15 != -4096 )
    {
      v13 = v12 & (v24 + v13);
      v25 = v24 + 1;
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( v9 == *v14 )
        goto LABEL_7;
      v24 = v25;
    }
    if ( v10 )
    {
      v23 = 64;
    }
    else
    {
      v21 = *(unsigned int *)(a1 + 512);
LABEL_27:
      v23 = 16 * v21;
    }
    v14 = (__int64 *)(v11 + v23);
    goto LABEL_7;
  }
  return 0;
}
