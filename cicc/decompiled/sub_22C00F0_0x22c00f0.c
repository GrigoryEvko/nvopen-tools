// Function: sub_22C00F0
// Address: 0x22c00f0
//
void __fastcall sub_22C00F0(__int64 a1, __int64 a2, char a3, char a4, unsigned int a5)
{
  char v9; // al
  unsigned int v10; // r13d
  unsigned __int8 v11; // al
  unsigned __int64 v12; // rdi
  const void *v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int8 v15; // al
  int v16; // eax
  bool v17; // cc
  unsigned __int8 v18; // al

  if ( sub_AAF760(a2) )
  {
    if ( *(_BYTE *)a1 == 6 )
      return;
    goto LABEL_3;
  }
  v9 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 1 )
  {
    v9 = 5;
    goto LABEL_28;
  }
  if ( v9 != 5 && !a3 )
  {
    if ( v9 == 4 )
      goto LABEL_9;
    v9 = 4;
LABEL_28:
    *(_BYTE *)(a1 + 1) = 0;
    *(_BYTE *)a1 = v9;
    v16 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a2 + 8) = 0;
    *(_DWORD *)(a1 + 16) = v16;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 16);
    *(_DWORD *)(a2 + 24) = 0;
    return;
  }
  v17 = (unsigned __int8)(v9 - 4) <= 1u;
  v9 = 5;
  if ( !v17 )
    goto LABEL_28;
LABEL_9:
  v10 = *(_DWORD *)(a1 + 16);
  *(_BYTE *)a1 = v9;
  if ( v10 > 0x40 )
  {
    if ( !sub_C43C50(a1 + 8, (const void **)a2) )
    {
      if ( !a4 || (v11 = *(_BYTE *)(a1 + 1) + 1, *(_BYTE *)(a1 + 1) = v11, v11 <= a5) )
      {
LABEL_13:
        v12 = *(_QWORD *)(a1 + 8);
        if ( v12 )
          j_j___libc_free_0_0(v12);
        goto LABEL_15;
      }
LABEL_3:
      sub_22C0090((unsigned __int8 *)a1);
      *(_BYTE *)a1 = 6;
      return;
    }
    goto LABEL_21;
  }
  v13 = *(const void **)a2;
  if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a2 )
  {
LABEL_21:
    if ( *(_DWORD *)(a1 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(a1 + 24) == *(_QWORD *)(a2 + 16) )
        return;
    }
    else if ( sub_C43C50(a1 + 24, (const void **)(a2 + 16)) )
    {
      return;
    }
    if ( a4 )
    {
      v15 = *(_BYTE *)(a1 + 1) + 1;
      *(_BYTE *)(a1 + 1) = v15;
      if ( v15 > a5 )
        goto LABEL_3;
    }
    if ( v10 > 0x40 )
      goto LABEL_13;
    goto LABEL_15;
  }
  if ( !a4 )
    goto LABEL_16;
  v18 = *(_BYTE *)(a1 + 1) + 1;
  *(_BYTE *)(a1 + 1) = v18;
  if ( a5 < v18 )
    goto LABEL_3;
LABEL_15:
  v13 = *(const void **)a2;
LABEL_16:
  *(_QWORD *)(a1 + 8) = v13;
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  if ( *(_DWORD *)(a1 + 32) > 0x40u )
  {
    v14 = *(_QWORD *)(a1 + 24);
    if ( v14 )
      j_j___libc_free_0_0(v14);
  }
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 24) = 0;
}
