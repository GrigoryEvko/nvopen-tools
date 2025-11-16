// Function: sub_D97FD0
// Address: 0xd97fd0
//
__int64 __fastcall sub_D97FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 result; // rax
  __int64 v7; // r9
  unsigned __int64 v9; // rcx
  char v10; // r10
  __int64 v11; // r11
  int v12; // edx
  unsigned int v13; // edi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rdx
  char v17; // dl
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rsi
  int v21; // esi
  int v22; // ebx
  __int64 v23; // [rsp+8h] [rbp-18h]

  result = a1;
  v7 = 4LL * a6;
  v9 = v7 | a4 & 0xFFFFFFFFFFFFFFFBLL;
  v10 = *(_BYTE *)(a2 + 8) & 1;
  if ( v10 )
  {
    v11 = a2 + 16;
    v12 = 3;
  }
  else
  {
    v11 = *(_QWORD *)(a2 + 16);
    v19 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v19 )
      goto LABEL_14;
    v12 = v19 - 1;
  }
  v13 = v12 & (v9 ^ (v9 >> 9));
  v14 = v11 + 88LL * v13;
  v7 = *(_QWORD *)v14;
  if ( v9 == *(_QWORD *)v14 )
    goto LABEL_4;
  v21 = 1;
  while ( v7 != -4 )
  {
    v22 = v21 + 1;
    v13 = v12 & (v21 + v13);
    v14 = v11 + 88LL * v13;
    v7 = *(_QWORD *)v14;
    if ( v9 == *(_QWORD *)v14 )
      goto LABEL_4;
    v21 = v22;
  }
  if ( v10 )
  {
    v20 = 352;
    goto LABEL_15;
  }
  v19 = *(unsigned int *)(a2 + 24);
LABEL_14:
  v20 = 88 * v19;
LABEL_15:
  v14 = v11 + v20;
LABEL_4:
  v15 = 352;
  if ( !v10 )
  {
    v16 = *(unsigned int *)(a2 + 24);
    v9 = 5 * v16;
    v15 = 88 * v16;
  }
  if ( v14 == v11 + v15 )
  {
    *(_BYTE *)(result + 80) = 0;
  }
  else
  {
    *(_QWORD *)result = *(_QWORD *)(v14 + 8);
    *(_QWORD *)(result + 8) = *(_QWORD *)(v14 + 16);
    *(_QWORD *)(result + 16) = *(_QWORD *)(v14 + 24);
    v17 = *(_BYTE *)(v14 + 32);
    *(_QWORD *)(result + 40) = 0x400000000LL;
    *(_BYTE *)(result + 24) = v17;
    *(_QWORD *)(result + 32) = result + 48;
    v18 = *(unsigned int *)(v14 + 48);
    if ( (_DWORD)v18 )
    {
      v23 = result;
      sub_D915C0(result + 32, v14 + 40, v18, v9, a2, v7);
      result = v23;
    }
    *(_BYTE *)(result + 80) = 1;
  }
  return result;
}
