// Function: sub_1CEFBE0
// Address: 0x1cefbe0
//
__int64 __fastcall sub_1CEFBE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // rdx
  int v9; // esi
  __int64 v10; // r14
  __int64 v11; // rbx
  unsigned int i; // r12d
  __int64 v13; // rdi
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+18h] [rbp-38h]
  int v19; // [rsp+20h] [rbp-30h]

  v2 = &v16;
  v4 = *(_QWORD *)(a1 + 160);
  v16 = 1;
  v17 = 0;
  v15 = v4;
  v18 = 0;
  v19 = 0;
  sub_1392B70((__int64)&v16, 0);
  if ( !v19 )
  {
    LODWORD(v18) = v18 + 1;
    BUG();
  }
  v5 = 1;
  v6 = 0;
  v7 = ((_WORD)v19 - 1) & 0x940;
  v8 = v17 + 8 * v7;
  v9 = *(_DWORD *)v8;
  if ( *(_DWORD *)v8 != 64 )
  {
    while ( v9 != -1 )
    {
      if ( v9 == -2 && !v6 )
        v6 = v8;
      v7 = (v19 - 1) & (unsigned int)(v5 + v7);
      v8 = v17 + 8LL * (unsigned int)v7;
      v9 = *(_DWORD *)v8;
      if ( *(_DWORD *)v8 == 64 )
        goto LABEL_3;
      v5 = (unsigned int)(v5 + 1);
    }
    if ( v6 )
      v8 = v6;
  }
LABEL_3:
  LODWORD(v18) = v18 + 1;
  if ( *(_DWORD *)v8 != -1 )
    --HIDWORD(v18);
  *(_QWORD *)v8 = 0x2000000040LL;
  if ( v15 )
  {
    v8 = *(unsigned int *)(v15 + 40);
    if ( (_DWORD)v8 )
      v2 = (__int64 *)(v15 + 24);
  }
  v10 = a2 + 72;
  v11 = *(_QWORD *)(a2 + 80);
  for ( i = 0; v10 != v11; i |= sub_394CF20(v13 - 24, v2, v8, v7, v6, v5) )
  {
    v13 = v11;
    v11 = *(_QWORD *)(v11 + 8);
  }
  j___libc_free_0(v17);
  return i;
}
