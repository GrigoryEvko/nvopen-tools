// Function: sub_1F04590
// Address: 0x1f04590
//
char __fastcall sub_1F04590(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v7; // rdi
  unsigned int v8; // r10d
  unsigned __int64 v9; // rcx
  int v10; // ebx
  bool v11; // di
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // r12
  unsigned __int64 v15; // r12
  unsigned int v16; // r10d
  __int64 v17; // r12
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  int i; // r15d

  v4 = *(unsigned int *)(a3 + 24);
  if ( !(_DWORD)v4 )
    return v4;
  v7 = a4 >> 2;
  v8 = (v4 - 1) & (37 * a4);
  v9 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = 1;
  v11 = !(v7 & 1);
  v12 = *(_QWORD *)(a3 + 8);
  while ( 1 )
  {
    v13 = v12 + 16LL * v8;
    v14 = *(_QWORD *)v13;
    if ( v11 != !((*(__int64 *)v13 >> 2) & 1) )
    {
      if ( ((*(__int64 *)v13 >> 2) & 1) != 0 )
        goto LABEL_12;
      v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      goto LABEL_8;
    }
    v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v11 )
      break;
    if ( v15 == v9 )
      goto LABEL_13;
LABEL_8:
    if ( v15 == -8 )
      return v4;
LABEL_12:
    v16 = v10 + v8;
    ++v10;
    v8 = (v4 - 1) & v16;
  }
  if ( v15 != v9 )
    goto LABEL_12;
LABEL_13:
  v4 = v12 + 16 * v4;
  if ( v13 != v4 )
  {
    v17 = *(_QWORD *)(a3 + 32) + 32LL * *(unsigned int *)(v13 + 8);
    if ( *(_QWORD *)(a3 + 40) != v17 )
    {
      v18 = *(_QWORD **)(v17 + 8);
      v19 = (_QWORD *)(v17 + 8);
      for ( i = *(_DWORD *)(a3 + 60); v19 != v18; v18 = (_QWORD *)*v18 )
        LOBYTE(v4) = sub_1F044A0(a1, a2, v18[2], i);
    }
  }
  return v4;
}
