// Function: sub_1B43FA0
// Address: 0x1b43fa0
//
__int64 __fastcall sub_1B43FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v8; // rax
  unsigned int v9; // r8d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r11
  __int64 v14; // rdi
  __int64 v15; // rax
  char v16; // r9
  unsigned int v17; // ecx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v28; // [rsp+0h] [rbp-40h]
  int v29; // [rsp+8h] [rbp-38h]
  unsigned int v30; // [rsp+Ch] [rbp-34h]

  v8 = sub_157EBA0(a1);
  if ( !v8 )
    return 1;
  v29 = sub_15F4D60(v8);
  v28 = sub_157EBA0(a1);
  if ( !v29 )
    return 1;
  v9 = 0;
  while ( 1 )
  {
    v30 = v9;
    v10 = sub_15F4DF0(v28, v9);
    v11 = sub_157F280(v10);
    v13 = v12;
    v14 = v11;
    if ( v11 != v12 )
      break;
LABEL_26:
    v9 = v30 + 1;
    if ( v29 == v30 + 1 )
      return 1;
  }
  while ( 1 )
  {
    v15 = 0x17FFFFFFE8LL;
    v16 = *(_BYTE *)(v14 + 23) & 0x40;
    v17 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
    if ( v17 )
    {
      v18 = 24LL * *(unsigned int *)(v14 + 56) + 8;
      v19 = 0;
      do
      {
        v20 = v14 - 24LL * v17;
        if ( v16 )
          v20 = *(_QWORD *)(v14 - 8);
        if ( a1 == *(_QWORD *)(v20 + v18) )
        {
          v15 = 24 * v19;
          goto LABEL_12;
        }
        ++v19;
        v18 += 8;
      }
      while ( v17 != (_DWORD)v19 );
      v15 = 0x17FFFFFFE8LL;
    }
LABEL_12:
    if ( v16 )
      v21 = *(_QWORD *)(v14 - 8);
    else
      v21 = v14 - 24LL * v17;
    v22 = *(_QWORD *)(v21 + v15);
    v23 = 0x17FFFFFFE8LL;
    if ( v17 )
    {
      v24 = 0;
      do
      {
        if ( a2 == *(_QWORD *)(v21 + 24LL * *(unsigned int *)(v14 + 56) + 8 * v24 + 8) )
        {
          v23 = 24 * v24;
          goto LABEL_19;
        }
        ++v24;
      }
      while ( v17 != (_DWORD)v24 );
      v23 = 0x17FFFFFFE8LL;
    }
LABEL_19:
    v25 = *(_QWORD *)(v21 + v23);
    if ( v22 != v25 && (a3 == v22 || a4 == v25) )
      return 0;
    v26 = *(_QWORD *)(v14 + 32);
    if ( !v26 )
      BUG();
    v14 = 0;
    if ( *(_BYTE *)(v26 - 8) == 77 )
      v14 = v26 - 24;
    if ( v13 == v14 )
      goto LABEL_26;
  }
}
