// Function: sub_1BBCF10
// Address: 0x1bbcf10
//
__int64 __fastcall sub_1BBCF10(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r15
  __int64 v9; // r14
  unsigned int v10; // r9d
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned __int64 v15; // rax
  unsigned int v16; // r8d
  int v17; // r14d
  _DWORD *v18; // rax
  _DWORD *i; // rdx
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // rdi
  _DWORD *v24; // rax
  _DWORD *v25; // rax

  if ( (*(_BYTE *)(a4 + 23) & 0x40) != 0 )
  {
    v8 = **(_QWORD **)(a4 - 8);
    *(_DWORD *)(a5 + 8) = 0;
    if ( *(_BYTE *)(a4 + 16) != 86 )
    {
LABEL_3:
      v9 = *(unsigned int *)(*(_QWORD *)v8 + 32LL);
      goto LABEL_4;
    }
  }
  else
  {
    v8 = *(_QWORD *)(a4 - 24LL * (*(_DWORD *)(a4 + 20) & 0xFFFFFFF));
    *(_DWORD *)(a5 + 8) = 0;
    if ( *(_BYTE *)(a4 + 16) != 86 )
      goto LABEL_3;
  }
  v12 = sub_15F2050(a4);
  v13 = sub_1632FA0(v12);
  v14 = sub_1BBCE00(a1, *(_QWORD *)v8, v13);
  v9 = v14;
  if ( !v14 || *(_BYTE *)(v8 + 16) != 54 || sub_15F32D0(v8) || (*(_BYTE *)(v8 + 18) & 1) != 0 || !sub_1648CD0(v8, a3) )
    return 0;
LABEL_4:
  if ( v9 != a3 )
    return 0;
  v15 = *(unsigned int *)(a5 + 12);
  *(_DWORD *)(a5 + 8) = 0;
  v16 = a3;
  v17 = a3 + 1;
  if ( a3 > v15 )
  {
    sub_16CD150(a5, (const void *)(a5 + 16), a3, 4, a3, a6);
    v16 = a3;
  }
  v18 = *(_DWORD **)a5;
  *(_DWORD *)(a5 + 8) = a3;
  for ( i = &v18[a3]; i != v18; ++v18 )
    *v18 = v17;
  if ( !a3 )
    return 1;
  v20 = 0;
  v10 = 1;
  while ( 1 )
  {
    v21 = *(__int64 **)(a2 + 8 * v20);
    v22 = (*((_BYTE *)v21 + 23) & 0x40) != 0 ? (__int64 *)*(v21 - 1) : &v21[-3 * (*((_DWORD *)v21 + 5) & 0xFFFFFFF)];
    if ( v8 != *v22 )
      break;
    if ( *((_BYTE *)v21 + 16) == 83 )
    {
      v23 = v22[3];
      if ( *(_BYTE *)(v23 + 16) != 13 )
        break;
      v21 = *(__int64 **)(v23 + 24);
      if ( *(_DWORD *)(v23 + 32) > 0x40u )
        v21 = (__int64 *)*v21;
    }
    else
    {
      if ( *((_DWORD *)v21 + 16) != 1 )
        break;
      LODWORD(v21) = *(_DWORD *)v21[7];
    }
    if ( (_DWORD)v20 == (_DWORD)v21 )
    {
      v25 = (_DWORD *)(*(_QWORD *)a5 + 4 * v20);
      if ( v17 != *v25 )
        break;
      *v25 = v20;
    }
    else
    {
      if ( v16 <= (unsigned int)v21 )
        break;
      v24 = (_DWORD *)(*(_QWORD *)a5 + 4LL * (unsigned int)v21);
      if ( v17 != *v24 )
        break;
      *v24 = v20;
      v10 = 0;
    }
    if ( (unsigned int)a3 == ++v20 )
      return v10;
  }
  if ( (unsigned int)a3 > (unsigned int)v20 )
  {
    *(_DWORD *)(a5 + 8) = 0;
    return 0;
  }
  return v10;
}
