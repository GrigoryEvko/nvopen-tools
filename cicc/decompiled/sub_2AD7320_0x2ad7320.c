// Function: sub_2AD7320
// Address: 0x2ad7320
//
void __fastcall sub_2AD7320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rbx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  char v15; // dl
  __int64 v16[3]; // [rsp+0h] [rbp-40h] BYREF
  char v17; // [rsp+18h] [rbp-28h]

  v6 = *(_QWORD *)(a1 + 104);
LABEL_2:
  v7 = *(_QWORD *)(v6 - 32);
  if ( !*(_BYTE *)(v6 - 8) )
  {
    *(_QWORD *)(v6 - 24) = v7;
    v8 = 0;
    v9 = v7;
    v10 = 1;
    *(_QWORD *)(v6 - 16) = 0;
    *(_BYTE *)(v6 - 8) = 1;
    if ( *(_BYTE *)(v7 + 8) )
    {
LABEL_12:
      v14 = v7;
      while ( 1 )
      {
        v10 = *(unsigned int *)(v14 + 88);
        if ( (_DWORD)v10 )
          break;
        v14 = *(_QWORD *)(v14 + 48);
        if ( !v14 )
        {
          v10 = 0;
          if ( v7 != v9 )
            goto LABEL_5;
          goto LABEL_16;
        }
      }
    }
    goto LABEL_4;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v6 - 24);
      v10 = 1;
      v8 = *(_QWORD *)(v6 - 16);
      if ( *(_BYTE *)(v7 + 8) )
        goto LABEL_12;
LABEL_4:
      if ( v7 == v9 )
      {
LABEL_16:
        if ( v8 == v10 )
        {
          *(_QWORD *)(a1 + 104) -= 32LL;
          v6 = *(_QWORD *)(a1 + 104);
          if ( v6 == *(_QWORD *)(a1 + 96) )
            return;
          goto LABEL_2;
        }
      }
LABEL_5:
      v11 = v8 + 1;
      *(_QWORD *)(v6 - 16) = v8 + 1;
      if ( *(_BYTE *)(v9 + 8) )
        break;
      v12 = *(_QWORD *)(v9 + 112);
      if ( *(_BYTE *)(a1 + 28) )
        goto LABEL_7;
LABEL_22:
      sub_C8CC70(a1, v12, v11, v10, a5, a6);
      if ( v15 )
        goto LABEL_23;
    }
    while ( 1 )
    {
      v11 = *(unsigned int *)(v9 + 88);
      if ( (_DWORD)v11 )
        break;
      v9 = *(_QWORD *)(v9 + 48);
      if ( !v9 )
        BUG();
    }
    v12 = *(_QWORD *)(*(_QWORD *)(v9 + 80) + 8LL * (unsigned int)v8);
    if ( !*(_BYTE *)(a1 + 28) )
      goto LABEL_22;
LABEL_7:
    v13 = *(_QWORD **)(a1 + 8);
    v10 = *(unsigned int *)(a1 + 20);
    v11 = (__int64)&v13[v10];
    if ( v13 == (_QWORD *)v11 )
      break;
    while ( v12 != *v13 )
    {
      if ( (_QWORD *)v11 == ++v13 )
        goto LABEL_24;
    }
  }
LABEL_24:
  if ( (unsigned int)v10 >= *(_DWORD *)(a1 + 16) )
    goto LABEL_22;
  *(_DWORD *)(a1 + 20) = v10 + 1;
  *(_QWORD *)v11 = v12;
  ++*(_QWORD *)a1;
LABEL_23:
  v16[0] = v12;
  v17 = 0;
  sub_2AC67A0((unsigned __int64 *)(a1 + 96), (__int64)v16);
}
