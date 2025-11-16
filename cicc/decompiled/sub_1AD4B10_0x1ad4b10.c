// Function: sub_1AD4B10
// Address: 0x1ad4b10
//
__int64 __fastcall sub_1AD4B10(__int64 a1, _QWORD *a2, int a3)
{
  unsigned __int64 v5; // r14
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // rax
  unsigned int *v14; // rax
  __int64 v15; // rdx
  int v16; // edi
  unsigned int v17; // esi
  __int64 v18; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // r12
  __int64 v24; // r15
  __int64 v25; // r12
  __int64 v26; // rax

  v5 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = *(_BYTE *)(v5 + 23);
  if ( (*a2 & 4) == 0 )
  {
    if ( v6 < 0 )
    {
      v20 = sub_1648A40(*a2 & 0xFFFFFFFFFFFFFFF8LL);
      v22 = v20 + v21;
      if ( *(char *)(v5 + 23) < 0 )
        v22 -= sub_1648A40(v5);
      v23 = v22 >> 4;
      if ( (_DWORD)v23 )
      {
        v24 = 0;
        v25 = 16LL * (unsigned int)v23;
        do
        {
          v26 = 0;
          if ( *(char *)(v5 + 23) < 0 )
            v26 = sub_1648A40(v5);
          v14 = (unsigned int *)(v24 + v26);
          v15 = *(_QWORD *)v14;
          if ( a3 == *(_DWORD *)(*(_QWORD *)v14 + 8LL) )
            goto LABEL_11;
          v24 += 16;
        }
        while ( v25 != v24 );
      }
    }
LABEL_13:
    *(_BYTE *)(a1 + 24) = 0;
    return a1;
  }
  if ( v6 >= 0 )
    goto LABEL_13;
  v7 = sub_1648A40(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  v9 = v7 + v8;
  if ( *(char *)(v5 + 23) < 0 )
    v9 -= sub_1648A40(v5);
  v10 = v9 >> 4;
  if ( !(_DWORD)v10 )
    goto LABEL_13;
  v11 = 0;
  v12 = 16LL * (unsigned int)v10;
  while ( 1 )
  {
    v13 = 0;
    if ( *(char *)(v5 + 23) < 0 )
      v13 = sub_1648A40(v5);
    v14 = (unsigned int *)(v11 + v13);
    v15 = *(_QWORD *)v14;
    if ( a3 == *(_DWORD *)(*(_QWORD *)v14 + 8LL) )
      break;
    v11 += 16;
    if ( v11 == v12 )
      goto LABEL_13;
  }
LABEL_11:
  v16 = *(_DWORD *)(v5 + 20);
  v17 = v14[3];
  v18 = v14[2];
  *(_BYTE *)(a1 + 24) = 1;
  *(_QWORD *)a1 = 24 * v18 - 24LL * (v16 & 0xFFFFFFF) + v5;
  *(_QWORD *)(a1 + 16) = v15;
  *(_QWORD *)(a1 + 8) = 0xAAAAAAAAAAAAAAABLL * ((24LL * v17 - 24 * v18) >> 3);
  return a1;
}
