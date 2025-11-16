// Function: sub_28B4EA0
// Address: 0x28b4ea0
//
__int64 __fastcall sub_28B4EA0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 *v3; // r10
  __int64 *v4; // r11
  __int64 *v5; // r9
  __int64 *v6; // rcx
  __int64 *v7; // rax
  unsigned int v8; // eax
  int v9; // edx
  unsigned int v10; // ebx
  int v11; // ebx
  __int64 *v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // r8d
  int v16; // eax
  int v17; // eax
  __int64 result; // rax
  __int64 *v19; // rax
  __int64 v20; // r9
  __int64 v21; // rdx
  int v22; // eax

  v2 = *a2;
  v3 = (__int64 *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = 0;
  v4 = a2 + 1;
  v5 = (__int64 *)(a1 + 24);
  v6 = (__int64 *)(a1 + 88);
  *(_QWORD *)(a1 + 16) = 1;
  *(_QWORD *)a1 = v2;
  v7 = (__int64 *)(a1 + 24);
  do
  {
    if ( v7 )
      *v7 = -4096;
    ++v7;
  }
  while ( v7 != v6 );
  v8 = *(_DWORD *)(a1 + 16) & 1 | a2[2] & 0xFFFFFFFE;
  v9 = *(_DWORD *)(a1 + 20);
  v10 = *(_DWORD *)(a1 + 16) & 0xFFFFFFFE | a2[2] & 1;
  *(_DWORD *)(a1 + 16) = v8;
  *((_DWORD *)a2 + 4) = v10;
  v11 = *((_DWORD *)a2 + 5);
  *((_DWORD *)a2 + 5) = v9;
  *(_DWORD *)(a1 + 20) = v11;
  if ( (v8 & 1) == 0 )
  {
    if ( (a2[2] & 1) == 0 )
    {
      v21 = a2[3];
      a2[3] = *(_QWORD *)(a1 + 24);
      v22 = *(_DWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 24) = v21;
      LODWORD(v21) = *((_DWORD *)a2 + 8);
      *((_DWORD *)a2 + 8) = v22;
      *(_DWORD *)(a1 + 32) = v21;
      goto LABEL_11;
    }
    v12 = (__int64 *)(a1 + 24);
    v5 = a2 + 3;
    v4 = (__int64 *)(a1 + 8);
    v3 = a2 + 1;
    goto LABEL_8;
  }
  v12 = a2 + 3;
  if ( (a2[2] & 1) == 0 )
  {
LABEL_8:
    *((_BYTE *)v4 + 8) |= 1u;
    v13 = v4[2];
    v14 = 0;
    v15 = *((_DWORD *)v4 + 6);
    do
    {
      v12[v14] = v5[v14];
      ++v14;
    }
    while ( v14 != 8 );
    *((_BYTE *)v3 + 8) &= ~1u;
    v3[2] = v13;
    *((_DWORD *)v3 + 6) = v15;
    goto LABEL_11;
  }
  v19 = (__int64 *)(a1 + 24);
  do
  {
    v20 = *v19;
    *v19++ = *v12;
    *v12++ = v20;
  }
  while ( v19 != v6 );
LABEL_11:
  *(_BYTE *)(a1 + 88) = *((_BYTE *)a2 + 88);
  *(_DWORD *)(a1 + 92) = *((_DWORD *)a2 + 23);
  *(_QWORD *)(a1 + 96) = a2[12];
  *(_QWORD *)(a1 + 104) = a2[13];
  *(_DWORD *)(a1 + 112) = *((_DWORD *)a2 + 28);
  v16 = *((_DWORD *)a2 + 32);
  *((_DWORD *)a2 + 32) = 0;
  *(_DWORD *)(a1 + 128) = v16;
  *(_QWORD *)(a1 + 120) = a2[15];
  *(_QWORD *)(a1 + 136) = a2[17];
  *(_QWORD *)(a1 + 144) = a2[18];
  *(_DWORD *)(a1 + 152) = *((_DWORD *)a2 + 38);
  v17 = *((_DWORD *)a2 + 42);
  *((_DWORD *)a2 + 42) = 0;
  *(_DWORD *)(a1 + 168) = v17;
  *(_QWORD *)(a1 + 160) = a2[20];
  *(_DWORD *)(a1 + 176) = *((_DWORD *)a2 + 44);
  result = a2[23];
  *(_QWORD *)(a1 + 184) = result;
  return result;
}
