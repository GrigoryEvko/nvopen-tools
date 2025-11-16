// Function: sub_359F080
// Address: 0x359f080
//
void __fastcall sub_359F080(__int64 *a1, __int64 a2, char a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r9
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r13
  int v12; // r14d
  int v13; // esi
  __int64 v14; // rdi
  unsigned __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // ecx
  int *v21; // r10
  int v22; // edi
  __int64 v23; // rdi
  int v24; // r10d
  int v27; // [rsp+18h] [rbp-48h]
  int v29[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v8 = *(_QWORD *)(a2 + 32);
  v27 = a5;
  v9 = v8 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v9 != v8 )
  {
    v10 = *(_QWORD *)(a2 + 32);
    v11 = a6 + 32LL * a4;
    while ( 1 )
    {
      if ( *(_BYTE *)v10 )
        goto LABEL_5;
      v13 = *(_DWORD *)(v10 + 8);
      if ( v13 >= 0 )
        goto LABEL_5;
      v29[0] = *(_DWORD *)(v10 + 8);
      v14 = a1[3];
      if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
      {
        v12 = sub_2EC06C0(
                v14,
                *(_QWORD *)(*(_QWORD *)(v14 + 56) + 16LL * (v13 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                byte_3F871B3,
                0,
                a5,
                v8);
        sub_2EAB0C0(v10, v12);
        *sub_2FFAE70(v11, v29) = v12;
        if ( a3 )
          sub_35988E0(v29[0], v12, a1[6], a1[3], a1[5], v8);
        goto LABEL_5;
      }
      v15 = sub_2EBEE10(v14, v13);
      v16 = sub_3598DB0(*a1, v15);
      if ( v16 == -1 || v27 <= v16 )
        v17 = v11;
      else
        v17 = a6 + 32LL * (a4 - v27 + v16);
      v18 = *(_QWORD *)(v17 + 8);
      v19 = *(unsigned int *)(v17 + 24);
      if ( !(_DWORD)v19 )
        goto LABEL_5;
      a5 = (unsigned int)(v19 - 1);
      v20 = a5 & (37 * v29[0]);
      v21 = (int *)(v18 + 8LL * v20);
      v22 = *v21;
      if ( v29[0] != *v21 )
      {
        v24 = 1;
        while ( v22 != -1 )
        {
          v8 = (unsigned int)(v24 + 1);
          v20 = a5 & (v24 + v20);
          v21 = (int *)(v18 + 8LL * v20);
          v22 = *v21;
          if ( v29[0] == *v21 )
            goto LABEL_14;
          v24 = v8;
        }
        goto LABEL_5;
      }
LABEL_14:
      if ( v21 == (int *)(v18 + 8 * v19) )
      {
LABEL_5:
        v10 += 40;
        if ( v9 == v10 )
          return;
      }
      else
      {
        v23 = v10;
        v10 += 40;
        sub_2EAB0C0(v23, v21[1]);
        if ( v9 == v10 )
          return;
      }
    }
  }
}
