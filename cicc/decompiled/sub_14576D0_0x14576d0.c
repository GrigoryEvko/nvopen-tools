// Function: sub_14576D0
// Address: 0x14576d0
//
__int64 __fastcall sub_14576D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  unsigned __int64 v6; // rcx
  char v9; // r8
  __int64 v10; // rax
  int v11; // r10d
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  int v21; // edx
  int v22; // r11d

  v6 = (4LL * a6) | a4 & 0xFFFFFFFFFFFFFFFBLL;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = a2 + 16;
    v11 = 3;
  }
  else
  {
    v19 = *(unsigned int *)(a2 + 24);
    v10 = *(_QWORD *)(a2 + 16);
    if ( !(_DWORD)v19 )
      goto LABEL_12;
    v11 = v19 - 1;
  }
  v12 = v11 & (v6 ^ (v6 >> 9));
  v13 = (__int64 *)(v10 + 104LL * v12);
  v14 = *v13;
  if ( v6 == *v13 )
    goto LABEL_4;
  v21 = 1;
  while ( v14 != -4 )
  {
    v22 = v21 + 1;
    v12 = v11 & (v21 + v12);
    v13 = (__int64 *)(v10 + 104LL * v12);
    v14 = *v13;
    if ( v6 == *v13 )
      goto LABEL_4;
    v21 = v22;
  }
  if ( v9 )
  {
    v20 = 416;
    goto LABEL_13;
  }
  v19 = *(unsigned int *)(a2 + 24);
LABEL_12:
  v20 = 104 * v19;
LABEL_13:
  v13 = (__int64 *)(v10 + v20);
LABEL_4:
  v15 = 416;
  if ( !v9 )
    v15 = 104LL * *(unsigned int *)(a2 + 24);
  if ( v13 == (__int64 *)(v15 + v10) )
  {
    *(_BYTE *)(a1 + 96) = 0;
  }
  else
  {
    *(_BYTE *)(a1 + 96) = 1;
    v16 = v13[1];
    v17 = v13 + 4;
    *(_QWORD *)a1 = v16;
    *(_QWORD *)(a1 + 8) = *(v17 - 2);
    *(_BYTE *)(a1 + 16) = *((_BYTE *)v17 - 8);
    sub_16CCCB0(a1 + 24, a1 + 64, v17);
  }
  return a1;
}
