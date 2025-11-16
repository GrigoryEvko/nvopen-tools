// Function: sub_2EF06E0
// Address: 0x2ef06e0
//
__int64 (__fastcall *__fastcall sub_2EF06E0(__int64 a1, char *a2, __int64 a3))(_QWORD *, _QWORD *, __int64)
{
  __int64 v5; // rdi
  void *v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // edi
  __int64 v9; // r8
  unsigned int v10; // r9d
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 i; // rsi
  __int16 v16; // dx
  __int64 v17; // r13
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rsi
  _BYTE *v22; // rax
  int v24; // ecx
  int v25; // edx
  int v26; // r10d
  _QWORD v27[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_2EF03A0(a1, a2, *(_QWORD *)(a3 + 24));
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xEu )
  {
    sub_CB6200(v5, "- instruction: ", 0xFu);
  }
  else
  {
    qmemcpy(v6, "- instruction: ", 15);
    *(_QWORD *)(v5 + 32) += 15LL;
  }
  v7 = *(_QWORD *)(a1 + 656);
  if ( v7 )
  {
    v8 = *(_DWORD *)(v7 + 144);
    v9 = *(_QWORD *)(v7 + 128);
    if ( v8 )
    {
      v10 = v8 - 1;
      v11 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v12 = *(_QWORD *)(v9 + 16LL * v11);
      if ( a3 == v12 )
      {
LABEL_6:
        v13 = a3;
        v14 = a3;
        if ( (*(_DWORD *)(a3 + 44) & 4) != 0 )
        {
          do
            v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v14 + 44) & 4) != 0 );
        }
        if ( (*(_DWORD *)(a3 + 44) & 8) != 0 )
        {
          do
            v13 = *(_QWORD *)(v13 + 8);
          while ( (*(_BYTE *)(v13 + 44) & 8) != 0 );
        }
        for ( i = *(_QWORD *)(v13 + 8); i != v14; v14 = *(_QWORD *)(v14 + 8) )
        {
          v16 = *(_WORD *)(v14 + 68);
          if ( (unsigned __int16)(v16 - 14) > 4u && v16 != 24 )
            break;
        }
        v17 = *(_QWORD *)(a1 + 16);
        v18 = v10 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v19 = (__int64 *)(v9 + 16LL * v18);
        v20 = *v19;
        if ( *v19 != v14 )
        {
          v25 = 1;
          while ( v20 != -4096 )
          {
            v26 = v25 + 1;
            v18 = v10 & (v25 + v18);
            v19 = (__int64 *)(v9 + 16LL * v18);
            v20 = *v19;
            if ( v14 == *v19 )
              goto LABEL_15;
            v25 = v26;
          }
          v19 = (__int64 *)(v9 + 16LL * v8);
        }
LABEL_15:
        v21 = *(_QWORD *)(a1 + 16);
        v27[0] = v19[1];
        sub_2FAD600(v27, v21);
        v22 = *(_BYTE **)(v17 + 32);
        if ( (unsigned __int64)v22 >= *(_QWORD *)(v17 + 24) )
        {
          sub_CB5D20(v17, 9);
        }
        else
        {
          *(_QWORD *)(v17 + 32) = v22 + 1;
          *v22 = 9;
        }
      }
      else
      {
        v24 = 1;
        while ( v12 != -4096 )
        {
          v11 = v10 & (v24 + v11);
          v12 = *(_QWORD *)(v9 + 16LL * v11);
          if ( a3 == v12 )
            goto LABEL_6;
          ++v24;
        }
      }
    }
  }
  return sub_2E91850(a3, *(_QWORD *)(a1 + 16), 1u, 0, 0, 1, 0);
}
