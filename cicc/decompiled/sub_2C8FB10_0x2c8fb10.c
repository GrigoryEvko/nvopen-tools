// Function: sub_2C8FB10
// Address: 0x2c8fb10
//
__int64 __fastcall sub_2C8FB10(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v4; // rax
  __int64 v5; // r9
  _DWORD *v6; // r8
  __int64 v7; // rcx
  _DWORD *v8; // rdx
  int v9; // esi
  __int64 v10; // r14
  __int64 v11; // rbx
  unsigned int i; // r12d
  __int64 v13; // rdi
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+18h] [rbp-38h]
  unsigned int v19; // [rsp+20h] [rbp-30h]

  v2 = &v16;
  v4 = *(_QWORD *)(a1 + 176);
  v16 = 1;
  v17 = 0;
  v15 = v4;
  v18 = 0;
  v19 = 0;
  sub_A09770((__int64)&v16, 0);
  if ( !v19 )
  {
    LODWORD(v18) = v18 + 1;
    BUG();
  }
  v5 = 1;
  v6 = 0;
  v7 = ((_WORD)v19 - 1) & 0x940;
  v8 = (_DWORD *)(v17 + 8 * v7);
  v9 = *v8;
  if ( *v8 != 64 )
  {
    while ( v9 != -1 )
    {
      if ( v9 == -2 && !v6 )
        v6 = v8;
      v7 = (v19 - 1) & ((_DWORD)v5 + (_DWORD)v7);
      v8 = (_DWORD *)(v17 + 8LL * (unsigned int)v7);
      v9 = *v8;
      if ( *v8 == 64 )
        goto LABEL_3;
      v5 = (unsigned int)(v5 + 1);
    }
    if ( v6 )
      v8 = v6;
  }
LABEL_3:
  LODWORD(v18) = v18 + 1;
  if ( *v8 != -1 )
    --HIDWORD(v18);
  *(_QWORD *)v8 = 0x2000000040LL;
  if ( v15 && *(_DWORD *)(v15 + 40) )
    v2 = (__int64 *)(v15 + 24);
  v10 = a2 + 72;
  v11 = *(_QWORD *)(a2 + 80);
  for ( i = 0; v10 != v11; i |= sub_3193470(v13 - 24, v2, 0, v7, v6, v5) )
  {
    v13 = v11;
    v11 = *(_QWORD *)(v11 + 8);
  }
  sub_C7D6A0(v17, 8LL * v19, 4);
  return i;
}
