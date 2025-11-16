// Function: sub_35845F0
// Address: 0x35845f0
//
__int64 __fastcall sub_35845F0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v4; // r13
  _QWORD *v7; // rcx
  unsigned int *v8; // rsi
  unsigned __int64 v9; // rdi
  unsigned __int64 v11; // r8
  _DWORD *v12; // r9
  _DWORD *v13; // rdx
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // edx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 (__fastcall **v21)(); // rax
  _DWORD v22[10]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a4;
  v22[1] = a4;
  v7 = *(_QWORD **)(a2 + 168);
  v8 = v22;
  v22[0] = a3;
  if ( v7 && (v9 = v7[1], v11 = a3 | (unsigned __int64)(v4 << 32), (v12 = *(_DWORD **)(*v7 + 8 * (v11 % v9))) != 0) )
  {
    v13 = *(_DWORD **)v12;
    v14 = *(_QWORD *)(*(_QWORD *)v12 + 24LL);
    while ( 1 )
    {
      if ( v11 == v14 && a3 == v13[2] && (_DWORD)v4 == v13[3] )
      {
        v8 = (unsigned int *)(*(_QWORD *)v12 + 16LL);
        if ( !*(_QWORD *)v12 )
          v8 = v22;
        goto LABEL_12;
      }
      if ( !*(_QWORD *)v13 )
        break;
      v14 = *(_QWORD *)(*(_QWORD *)v13 + 24LL);
      v12 = v13;
      if ( v11 % v9 != v14 % v9 )
        break;
      v13 = *(_DWORD **)v13;
    }
    v15 = *(_QWORD *)(a2 + 88);
    v8 = v22;
    v16 = a2 + 80;
    if ( !v15 )
      goto LABEL_26;
  }
  else
  {
LABEL_12:
    v15 = *(_QWORD *)(a2 + 88);
    v16 = a2 + 80;
    if ( !v15 )
      goto LABEL_26;
  }
  v17 = *v8;
  v18 = v16;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v15 + 32) < v17 )
      {
        v15 = *(_QWORD *)(v15 + 24);
        goto LABEL_18;
      }
      if ( *(_DWORD *)(v15 + 32) == v17 && *(_DWORD *)(v15 + 36) < v8[1] )
        break;
      v18 = v15;
      v15 = *(_QWORD *)(v15 + 16);
      if ( !v15 )
        goto LABEL_19;
    }
    v15 = *(_QWORD *)(v15 + 24);
LABEL_18:
    ;
  }
  while ( v15 );
LABEL_19:
  if ( v16 == v18 || *(_DWORD *)(v18 + 32) > v17 || *(_DWORD *)(v18 + 32) == v17 && v8[1] < *(_DWORD *)(v18 + 36) )
  {
LABEL_26:
    v21 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v21;
    *(_DWORD *)a1 = 0;
    return a1;
  }
  else
  {
    v19 = *(_QWORD *)(v18 + 40);
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v19;
    return a1;
  }
}
