// Function: sub_18E8200
// Address: 0x18e8200
//
unsigned __int64 __fastcall sub_18E8200(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r8
  int v5; // eax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r12
  __int64 *i; // rbx
  unsigned __int64 v17; // rax
  int v18; // edx
  int v19; // r10d

  if ( a3 == -1 )
  {
    v5 = *(unsigned __int8 *)(a2 + 16);
    if ( (_BYTE)v5 == 77 )
      goto LABEL_11;
  }
  else
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v3 = *(_QWORD *)(a2 - 8);
    else
      v3 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v3 + 24LL * a3) + 16LL) - 60) <= 0xCu )
      return *(_QWORD *)(v3 + 24LL * a3);
    v5 = *(unsigned __int8 *)(a2 + 16);
    if ( (_BYTE)v5 == 77 )
      return sub_157EBA0(*(_QWORD *)(v3 + 8LL * a3 + 24LL * *(unsigned int *)(a2 + 56) + 8));
  }
  v6 = (unsigned int)(v5 - 34);
  if ( (unsigned int)v6 > 0x36 )
    return a2;
  v7 = 0x40018000000001LL;
  if ( !_bittest64(&v7, v6) )
    return a2;
LABEL_11:
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(unsigned int *)(v8 + 48);
  if ( !(_DWORD)v9 )
    goto LABEL_26;
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)(v8 + 32);
  v12 = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v13 = (__int64 *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( v10 != *v13 )
  {
    v18 = 1;
    while ( v14 != -8 )
    {
      v19 = v18 + 1;
      v12 = (v9 - 1) & (v18 + v12);
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v10 == *v13 )
        goto LABEL_13;
      v18 = v19;
    }
LABEL_26:
    BUG();
  }
LABEL_13:
  if ( v13 == (__int64 *)(v11 + 16 * v9) )
    goto LABEL_26;
  v15 = 0x40018000000001LL;
  for ( i = *(__int64 **)(v13[1] + 8); ; i = (__int64 *)i[1] )
  {
    v17 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(*i) + 16) - 34;
    if ( (unsigned int)v17 > 0x36 || !_bittest64(&v15, v17) )
      break;
  }
  return sub_157EBA0(*i);
}
