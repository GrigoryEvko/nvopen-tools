// Function: sub_1B44430
// Address: 0x1b44430
//
__int64 __fastcall sub_1B44430(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int v11; // r10d
  __int64 v12; // rsi
  char v13; // di
  int v14; // r9d
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // [rsp+8h] [rbp-38h]

  result = sub_157F280(a1);
  if ( result != v6 )
  {
    v8 = v6;
    v9 = result;
    do
    {
      v10 = 0x17FFFFFFE8LL;
      v11 = *(_DWORD *)(v9 + 56);
      v12 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v13 = *(_BYTE *)(v9 + 23) & 0x40;
      v14 = v12;
      if ( (_DWORD)v12 )
      {
        v7 = v9 - 24LL * (unsigned int)v12;
        v15 = 24LL * v11 + 8;
        v16 = 0;
        do
        {
          v17 = v9 - 24LL * (unsigned int)v12;
          if ( v13 )
            v17 = *(_QWORD *)(v9 - 8);
          if ( a3 == *(_QWORD *)(v17 + v15) )
          {
            v10 = 24 * v16;
            goto LABEL_10;
          }
          ++v16;
          v15 += 8;
        }
        while ( (_DWORD)v12 != (_DWORD)v16 );
        v10 = 0x17FFFFFFE8LL;
      }
LABEL_10:
      if ( v13 )
      {
        v18 = *(_QWORD *)(v9 - 8);
        v19 = *(_QWORD *)(v18 + v10);
        if ( (_DWORD)v12 == v11 )
          goto LABEL_31;
      }
      else
      {
        v18 = v9 - 24LL * (unsigned int)v12;
        v19 = *(_QWORD *)(v18 + v10);
        if ( (_DWORD)v12 == v11 )
        {
LABEL_31:
          v29 = v19;
          sub_15F55D0(v9, v12, v18, v19, v7, (unsigned int)v12);
          v19 = v29;
          v14 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
        }
      }
      v20 = (v14 + 1) & 0xFFFFFFF;
      v21 = v20 | *(_DWORD *)(v9 + 20) & 0xF0000000;
      *(_DWORD *)(v9 + 20) = v21;
      if ( (v21 & 0x40000000) != 0 )
        v22 = *(_QWORD *)(v9 - 8);
      else
        v22 = v9 - 24 * v20;
      v23 = (__int64 *)(v22 + 24LL * (unsigned int)(v20 - 1));
      if ( *v23 )
      {
        v24 = v23[1];
        v25 = v23[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v25 = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = *(_QWORD *)(v24 + 16) & 3LL | v25;
      }
      *v23 = v19;
      if ( v19 )
      {
        v26 = *(_QWORD *)(v19 + 8);
        v23[1] = v26;
        if ( v26 )
        {
          v7 = (__int64)(v23 + 1);
          *(_QWORD *)(v26 + 16) = (unsigned __int64)(v23 + 1) | *(_QWORD *)(v26 + 16) & 3LL;
        }
        v23[2] = (v19 + 8) | v23[2] & 3;
        *(_QWORD *)(v19 + 8) = v23;
      }
      v27 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
        v28 = *(_QWORD *)(v9 - 8);
      else
        v28 = v9 - 24 * v27;
      *(_QWORD *)(v28 + 8LL * (unsigned int)(v27 - 1) + 24LL * *(unsigned int *)(v9 + 56) + 8) = a2;
      result = *(_QWORD *)(v9 + 32);
      if ( !result )
        BUG();
      v9 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v9 = result - 24;
    }
    while ( v8 != v9 );
  }
  return result;
}
