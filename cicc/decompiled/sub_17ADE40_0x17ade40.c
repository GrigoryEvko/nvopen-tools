// Function: sub_17ADE40
// Address: 0x17ade40
//
__int64 __fastcall sub_17ADE40(unsigned int a1, __int64 a2, unsigned __int8 a3)
{
  __int64 **v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  int v16; // [rsp+8h] [rbp-C8h]
  int v17; // [rsp+8h] [rbp-C8h]
  __int64 *v18; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+18h] [rbp-B8h]
  _BYTE s[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = **(__int64 ****)(*(_QWORD *)a2 + 16LL);
  v7 = sub_15A14F0(a1, v4, a3);
  if ( !v7 )
  {
    if ( a3 )
    {
      if ( a1 > 0x15 )
        v7 = sub_15A10B0((__int64)v4, 1.0);
      else
        v7 = sub_15A0680((__int64)v4, 1, 0);
    }
    else
    {
      v7 = sub_15A06D0(v4, (__int64)v4, v5, v6);
    }
  }
  v8 = *(_QWORD *)a2;
  v18 = (__int64 *)s;
  v9 = (__int64 *)s;
  v10 = *(_QWORD *)(v8 + 32);
  v19 = 0x1000000000LL;
  v11 = (unsigned int)v10;
  if ( (unsigned int)v10 > 0x10 )
  {
    v17 = v10;
    sub_16CD150((__int64)&v18, s, (unsigned int)v10, 8, v10, (int)&v18);
    v9 = v18;
    LODWORD(v10) = v17;
  }
  LODWORD(v19) = v10;
  if ( 8 * v11 )
  {
    v16 = v10;
    memset(v9, 0, 8 * v11);
    LODWORD(v10) = v16;
  }
  v12 = 0;
  if ( (_DWORD)v10 )
  {
    do
    {
      v13 = sub_15A0A60(a2, v12);
      if ( *(_BYTE *)(v13 + 16) == 9 )
        v13 = v7;
      v18[v12++] = v13;
    }
    while ( v11 != v12 );
  }
  v14 = sub_15A01B0(v18, (unsigned int)v19);
  if ( v18 != (__int64 *)s )
    _libc_free((unsigned __int64)v18);
  return v14;
}
