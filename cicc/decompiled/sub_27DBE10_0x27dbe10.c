// Function: sub_27DBE10
// Address: 0x27dbe10
//
__int64 __fastcall sub_27DBE10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned int v11; // r8d
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r9
  unsigned int v17; // edi
  __int64 v18; // rsi
  __int64 v19; // r10
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // esi
  int v25; // [rsp+Ch] [rbp-44h]
  __int64 v26; // [rsp+10h] [rbp-40h]

  result = sub_AA5930(a1);
  v26 = v7;
  if ( v7 != result )
  {
    v8 = result;
    do
    {
      v9 = *(_QWORD *)(v8 - 8);
      v10 = 0x1FFFFFFFE0LL;
      v11 = *(_DWORD *)(v8 + 72);
      v12 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
      if ( v12 )
      {
        v13 = 0;
        do
        {
          if ( a2 == *(_QWORD *)(v9 + 32LL * v11 + 8 * v13) )
          {
            v10 = 32 * v13;
            goto LABEL_8;
          }
          ++v13;
        }
        while ( v12 != (_DWORD)v13 );
        v10 = 0x1FFFFFFFE0LL;
      }
LABEL_8:
      v14 = *(_QWORD *)(v9 + v10);
      if ( *(_BYTE *)v14 > 0x1Cu )
      {
        v15 = *(unsigned int *)(a4 + 24);
        if ( (_DWORD)v15 )
        {
          v16 = *(_QWORD *)(a4 + 8);
          v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v18 = v16 + ((unsigned __int64)v17 << 6);
          v19 = *(_QWORD *)(v18 + 24);
          if ( v14 == v19 )
          {
LABEL_11:
            if ( v18 != v16 + (v15 << 6) )
              v14 = *(_QWORD *)(v18 + 56);
          }
          else
          {
            v24 = 1;
            while ( v19 != -4096 )
            {
              v17 = (v15 - 1) & (v24 + v17);
              v25 = v24 + 1;
              v18 = v16 + ((unsigned __int64)v17 << 6);
              v19 = *(_QWORD *)(v18 + 24);
              if ( v14 == v19 )
                goto LABEL_11;
              v24 = v25;
            }
          }
        }
      }
      if ( v12 == v11 )
      {
        sub_B48D90(v8);
        v9 = *(_QWORD *)(v8 - 8);
        v12 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
      }
      v20 = (v12 + 1) & 0x7FFFFFF;
      *(_DWORD *)(v8 + 4) = v20 | *(_DWORD *)(v8 + 4) & 0xF8000000;
      v21 = 32LL * (unsigned int)(v20 - 1) + v9;
      if ( *(_QWORD *)v21 )
      {
        v22 = *(_QWORD *)(v21 + 8);
        **(_QWORD **)(v21 + 16) = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = *(_QWORD *)(v21 + 16);
      }
      *(_QWORD *)v21 = v14;
      if ( v14 )
      {
        v23 = *(_QWORD *)(v14 + 16);
        *(_QWORD *)(v21 + 8) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = v21 + 8;
        *(_QWORD *)(v21 + 16) = v14 + 16;
        *(_QWORD *)(v14 + 16) = v21;
      }
      *(_QWORD *)(*(_QWORD *)(v8 - 8)
                + 32LL * *(unsigned int *)(v8 + 72)
                + 8LL * ((*(_DWORD *)(v8 + 4) & 0x7FFFFFFu) - 1)) = a3;
      result = *(_QWORD *)(v8 + 32);
      if ( !result )
        BUG();
      v8 = 0;
      if ( *(_BYTE *)(result - 24) == 84 )
        v8 = result - 24;
    }
    while ( v26 != v8 );
  }
  return result;
}
