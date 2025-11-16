// Function: sub_29D0010
// Address: 0x29d0010
//
unsigned __int8 *__fastcall sub_29D0010(_QWORD *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int8 *v8; // r12
  unsigned __int8 v9; // al
  __int64 v11; // rax
  __int64 v12; // rcx
  int v13; // eax
  int v14; // esi
  unsigned int v15; // edx
  unsigned __int8 **v16; // rax
  unsigned __int8 *v17; // rdi
  int i; // eax

  v8 = sub_BD3990(*((unsigned __int8 **)a2 - 4), (__int64)a2);
  v9 = *v8;
  if ( *v8 > 0x15u )
  {
    v11 = a1[6];
    if ( v11 == a1[7] )
      v11 = *(_QWORD *)(a1[9] - 8LL) + 512LL;
    v12 = *(_QWORD *)(v11 - 24);
    v13 = *(_DWORD *)(v11 - 8);
    if ( !v13 )
      BUG();
    v14 = v13 - 1;
    v15 = (v13 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v16 = (unsigned __int8 **)(v12 + 16LL * v15);
    v17 = *v16;
    if ( v8 != *v16 )
    {
      for ( i = 1; ; i = v6 )
      {
        if ( v17 == (unsigned __int8 *)-4096LL )
          BUG();
        v6 = (unsigned int)(i + 1);
        v15 = v14 & (i + v15);
        v16 = (unsigned __int8 **)(v12 + 16LL * v15);
        v17 = *v16;
        if ( v8 == *v16 )
          break;
      }
    }
    v8 = v16[1];
    v9 = *v8;
  }
  if ( v9 )
  {
    if ( v9 != 1 )
      return 0;
    v8 = (unsigned __int8 *)*((_QWORD *)v8 - 4);
    if ( *v8 )
      return 0;
  }
  if ( !(unsigned __int8)sub_29CFDA0(a1, a2, (__int64)v8, a3, v6, v7) )
    return 0;
  else
    return v8;
}
