// Function: sub_1FDEAD0
// Address: 0x1fdead0
//
__int64 __fastcall sub_1FDEAD0(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v5; // edx
  int v6; // r11d
  __int64 *v7; // r10
  __int64 v8; // rax
  __int64 v9; // r9
  unsigned int v10; // r8d
  unsigned __int64 v11; // rbx
  bool v12; // cl
  __int64 *v13; // rax
  __int64 v14; // r12
  unsigned __int64 v15; // r12
  unsigned int v17; // r8d

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = 1;
  v7 = 0;
  v8 = *a2;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = (v5 - 1) & (37 * v8);
  v11 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = !((v8 >> 2) & 1);
  while ( 1 )
  {
    v13 = (__int64 *)(v9 + 16LL * v10);
    v14 = *v13;
    if ( v12 != !((*v13 >> 2) & 1) )
      break;
    v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12 )
    {
      if ( v11 == v15 )
        goto LABEL_19;
      goto LABEL_8;
    }
    if ( v11 == v15 )
    {
LABEL_19:
      *a3 = v13;
      return 1;
    }
LABEL_15:
    v17 = v6 + v10;
    ++v6;
    v10 = (v5 - 1) & v17;
  }
  v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  if ( ((*v13 >> 2) & 1) != 0 )
    goto LABEL_15;
LABEL_8:
  if ( v15 != -8 )
  {
    if ( v15 == -16 && !v7 )
      v7 = (__int64 *)(v9 + 16LL * v10);
    goto LABEL_15;
  }
  if ( !v7 )
    v7 = (__int64 *)(v9 + 16LL * v10);
  *a3 = v7;
  return 0;
}
