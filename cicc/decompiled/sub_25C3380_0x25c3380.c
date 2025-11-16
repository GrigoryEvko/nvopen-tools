// Function: sub_25C3380
// Address: 0x25c3380
//
void __fastcall sub_25C3380(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // r12
  char v3; // dl
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 *v6; // r13
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // [rsp+0h] [rbp-80h]
  char v11; // [rsp+8h] [rbp-78h]
  _BYTE *v12; // [rsp+10h] [rbp-70h] BYREF
  __int64 v13; // [rsp+18h] [rbp-68h]
  _BYTE v14[96]; // [rsp+20h] [rbp-60h] BYREF

  v1 = a1;
  v2 = (__int64 *)(a1 + 16);
  v3 = *(_BYTE *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  v12 = v14;
  LODWORD(v1) = *(_DWORD *)(v1 + 24);
  v13 = 0x200000000LL;
  v10 = v4;
  v11 = v3;
  if ( (_DWORD)v1 )
    sub_25C2C90((__int64)&v12, v2);
  while ( 1 )
  {
    v6 = v2 - 2;
    if ( !sub_B445A0(v4, *(v2 - 14)) )
      break;
    v5 = (__int64)v2;
    *(v2 - 2) = *(v2 - 14);
    *((_BYTE *)v2 - 8) = *((_BYTE *)v2 - 104);
    v2 -= 12;
    sub_25C2C90(v5, v2);
    v4 = v10;
  }
  *v6 = v10;
  *((_BYTE *)v6 + 8) = v11;
  sub_25C2C90((__int64)v2, (__int64 *)&v12);
  v7 = (__int64)v12;
  v8 = (unsigned __int64)&v12[32 * (unsigned int)v13];
  if ( v12 != (_BYTE *)v8 )
  {
    do
    {
      v8 -= 32LL;
      if ( *(_DWORD *)(v8 + 24) > 0x40u )
      {
        v9 = *(_QWORD *)(v8 + 16);
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
      if ( *(_DWORD *)(v8 + 8) > 0x40u && *(_QWORD *)v8 )
        j_j___libc_free_0_0(*(_QWORD *)v8);
    }
    while ( v7 != v8 );
    v8 = (unsigned __int64)v12;
  }
  if ( (_BYTE *)v8 != v14 )
    _libc_free(v8);
}
