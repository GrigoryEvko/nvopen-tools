// Function: sub_1AD3A90
// Address: 0x1ad3a90
//
void __fastcall sub_1AD3A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 v15; // rsi
  __int64 v16; // r11
  __int64 v17; // rbx
  __int64 j; // r14
  __int64 v19; // rbx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // esi
  int v24; // ebx
  __int64 v25; // rdi
  __int64 i; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  char v29; // [rsp+18h] [rbp-38h]

  if ( (*(_DWORD *)(a3 + 8) & 0xFFFFFFFD) == 0 || !*(_QWORD *)a3 )
    return;
  if ( !a5 )
  {
    v29 = 0;
LABEL_22:
    v27 = 0;
    if ( !*(_DWORD *)(a2 + 16) )
      goto LABEL_8;
LABEL_23:
    v19 = *(_QWORD *)(a2 + 8);
    v20 = v19 + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6);
    if ( v19 != v20 )
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(v19 + 24);
        if ( v21 != -16 && v21 != -8 )
          break;
        v19 += 64;
        if ( v20 == v19 )
          goto LABEL_8;
      }
      while ( v20 != v19 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v19 + 24) + 16LL) == 78 )
        {
          v25 = *(_QWORD *)(v19 + 56);
          if ( v25 )
          {
            if ( *(_BYTE *)(v25 + 16) == 78 )
              sub_15F35F0(v25, v27, *(_QWORD *)a3);
          }
        }
        v19 += 64;
        if ( v19 == v20 )
          break;
        while ( 1 )
        {
          v22 = *(_QWORD *)(v19 + 24);
          if ( v22 != -16 && v22 != -8 )
            break;
          v19 += 64;
          if ( v20 == v19 )
            goto LABEL_8;
        }
      }
    }
    goto LABEL_8;
  }
  sub_1441B50((__int64)&v28, a5, a4, a6);
  v9 = *(_QWORD *)a3;
  if ( !v29 )
    goto LABEL_22;
  if ( v28 <= v9 )
    v9 = v28;
  v27 = v9;
  if ( *(_DWORD *)(a2 + 16) )
    goto LABEL_23;
LABEL_8:
  v10 = *(_QWORD *)(a1 + 80);
  for ( i = a1 + 72; i != v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    v11 = *(unsigned int *)(a2 + 24);
    v12 = v10 - 24;
    if ( !v10 )
      v12 = 0;
    if ( (_DWORD)v11 )
    {
      v13 = *(_QWORD *)(a2 + 8);
      v14 = (v11 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v15 = v13 + ((unsigned __int64)v14 << 6);
      v16 = *(_QWORD *)(v15 + 24);
      if ( v12 == v16 )
      {
LABEL_13:
        if ( v15 != v13 + (v11 << 6) )
        {
          v17 = *(_QWORD *)(v12 + 48);
          for ( j = v12 + 40; j != v17; v17 = *(_QWORD *)(v17 + 8) )
          {
            if ( !v17 )
              BUG();
            if ( *(_BYTE *)(v17 - 8) == 78 )
              sub_15F35F0(v17 - 24, *(_QWORD *)a3 - v27, *(_QWORD *)a3);
          }
        }
      }
      else
      {
        v23 = 1;
        while ( v16 != -8 )
        {
          v24 = v23 + 1;
          v14 = (v11 - 1) & (v23 + v14);
          v15 = v13 + ((unsigned __int64)v14 << 6);
          v16 = *(_QWORD *)(v15 + 24);
          if ( v12 == v16 )
            goto LABEL_13;
          v23 = v24;
        }
      }
    }
  }
}
