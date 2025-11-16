// Function: sub_19523F0
// Address: 0x19523f0
//
__int64 __fastcall sub_19523F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rcx
  unsigned int v11; // edi
  unsigned int v12; // eax
  __int64 v13; // r9
  int v14; // edx
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // r13
  __int64 v19; // r10
  __int64 v20; // rdx
  unsigned int v21; // ecx
  int v22; // eax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // [rsp+Ch] [rbp-44h]
  __int64 v31; // [rsp+10h] [rbp-40h]

  result = sub_157F280(a1);
  v31 = v8;
  if ( result != v8 )
  {
    v9 = result;
    do
    {
      v10 = 0x17FFFFFFE8LL;
      v11 = *(_DWORD *)(v9 + 56);
      v12 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      v13 = *(_BYTE *)(v9 + 23) & 0x40;
      v14 = v12;
      if ( v12 )
      {
        v15 = 24LL * v11 + 8;
        v16 = 0;
        do
        {
          v7 = v9 - 24LL * v12;
          if ( (_BYTE)v13 )
            v7 = *(_QWORD *)(v9 - 8);
          if ( a2 == *(_QWORD *)(v7 + v15) )
          {
            v10 = 24 * v16;
            goto LABEL_10;
          }
          ++v16;
          v15 += 8;
        }
        while ( v12 != (_DWORD)v16 );
        v10 = 0x17FFFFFFE8LL;
      }
LABEL_10:
      if ( (_BYTE)v13 )
      {
        v17 = *(_QWORD *)(v9 - 8);
      }
      else
      {
        v7 = 24LL * v12;
        v17 = v9 - v7;
      }
      v18 = *(_QWORD *)(v17 + v10);
      if ( *(_BYTE *)(v18 + 16) > 0x17u )
      {
        v10 = *(unsigned int *)(a4 + 24);
        if ( (_DWORD)v10 )
        {
          v13 = *(_QWORD *)(a4 + 8);
          v7 = ((_DWORD)v10 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v17 = v13 + 16 * v7;
          v19 = *(_QWORD *)v17;
          if ( v18 == *(_QWORD *)v17 )
          {
LABEL_15:
            v10 = v13 + 16 * v10;
            if ( v17 != v10 )
              v18 = *(_QWORD *)(v17 + 8);
          }
          else
          {
            v17 = 1;
            while ( v19 != -8 )
            {
              v7 = ((_DWORD)v10 - 1) & (unsigned int)(v17 + v7);
              v30 = v17 + 1;
              v17 = v13 + 16LL * (unsigned int)v7;
              v19 = *(_QWORD *)v17;
              if ( v18 == *(_QWORD *)v17 )
                goto LABEL_15;
              v17 = v30;
            }
          }
        }
      }
      if ( v11 == v12 )
      {
        sub_15F55D0(v9, v17, v12, v10, v7, v13);
        v14 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      }
      v20 = (v14 + 1) & 0xFFFFFFF;
      v21 = v20 - 1;
      v22 = v20 | *(_DWORD *)(v9 + 20) & 0xF0000000;
      *(_DWORD *)(v9 + 20) = v22;
      if ( (v22 & 0x40000000) != 0 )
        v23 = *(_QWORD *)(v9 - 8);
      else
        v23 = v9 - 24 * v20;
      v24 = (_QWORD *)(v23 + 24LL * v21);
      if ( *v24 )
      {
        v25 = v24[1];
        v26 = v24[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v26 = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = *(_QWORD *)(v25 + 16) & 3LL | v26;
      }
      *v24 = v18;
      if ( v18 )
      {
        v27 = *(_QWORD *)(v18 + 8);
        v24[1] = v27;
        if ( v27 )
          *(_QWORD *)(v27 + 16) = (unsigned __int64)(v24 + 1) | *(_QWORD *)(v27 + 16) & 3LL;
        v24[2] = (v18 + 8) | v24[2] & 3LL;
        *(_QWORD *)(v18 + 8) = v24;
      }
      v28 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
        v29 = *(_QWORD *)(v9 - 8);
      else
        v29 = v9 - 24 * v28;
      *(_QWORD *)(v29 + 8LL * (unsigned int)(v28 - 1) + 24LL * *(unsigned int *)(v9 + 56) + 8) = a3;
      result = *(_QWORD *)(v9 + 32);
      if ( !result )
        BUG();
      v9 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v9 = result - 24;
    }
    while ( v31 != v9 );
  }
  return result;
}
