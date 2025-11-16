// Function: sub_2E7AE30
// Address: 0x2e7ae30
//
__int64 __fastcall sub_2E7AE30(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // r8
  int v6; // edx
  unsigned __int64 v7; // rax
  int v8; // r15d
  unsigned __int64 v9; // rdx
  unsigned __int8 v10; // bl
  int v11; // r13d
  int v12; // r14d
  int v13; // ebx
  __int64 result; // rax
  int v15; // r9d
  unsigned __int64 v16; // rdx
  char v17; // r9
  int v18; // r8d
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  int v22; // [rsp+8h] [rbp-68h]
  unsigned __int8 v23; // [rsp+17h] [rbp-59h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  int v26; // [rsp+20h] [rbp-50h]
  __int64 v27; // [rsp+28h] [rbp-48h]
  int v28; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
    goto LABEL_7;
  if ( ((v4 >> 2) & 1) != 0 )
  {
    if ( ((v4 >> 2) & 1) != 0 )
    {
      v16 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      LODWORD(v4) = v4 & 0xFFFFFFF8 | 4;
      if ( v16 )
        v6 = *(_DWORD *)(v16 + 12);
      else
        v6 = 0;
      goto LABEL_8;
    }
LABEL_7:
    LODWORD(v4) = 4;
    v6 = 0;
    goto LABEL_8;
  }
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  if ( !v4 )
    goto LABEL_7;
  v5 = *(_QWORD *)(v4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = *(_DWORD *)(v5 + 8) >> 8;
LABEL_8:
  v26 = v4;
  v7 = *(_QWORD *)(a2 + 24);
  v28 = v6;
  v8 = *(unsigned __int16 *)(a2 + 32);
  LODWORD(v9) = -1;
  v27 = *(_QWORD *)(a2 + 8);
  if ( (v7 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v17 = *(_BYTE *)(a2 + 24) & 2;
    if ( (*(_BYTE *)(a2 + 24) & 6) == 2 || (*(_BYTE *)(a2 + 24) & 1) != 0 )
    {
      v21 = HIWORD(v7);
      if ( !v17 )
        v21 = HIDWORD(v7);
      v9 = (v21 + 7) >> 3;
    }
    else
    {
      v18 = (unsigned __int16)((unsigned int)v7 >> 8);
      v19 = HIWORD(v7);
      v20 = HIDWORD(*(_QWORD *)(a2 + 24));
      if ( v17 )
        LODWORD(v20) = v19;
      v9 = ((unsigned __int64)(unsigned int)(v18 * v20) + 7) >> 3;
    }
  }
  v10 = *(_BYTE *)(a2 + 37);
  v22 = v9;
  v11 = *(unsigned __int8 *)(a2 + 36);
  v23 = *(_BYTE *)(a2 + 34);
  v12 = v10 & 0xF;
  v24 = *(_QWORD *)(a2 + 72);
  v13 = v10 >> 4;
  result = sub_A777F0(0x58u, (__int64 *)(a1 + 128));
  if ( result )
  {
    v15 = v24;
    v25 = result;
    sub_2EAC440(result, v8, v22, v23, a3, v15, v26, v27, v28, v11, v12, v13);
    return v25;
  }
  return result;
}
