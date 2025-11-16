// Function: sub_25C3260
// Address: 0x25c3260
//
void __fastcall sub_25C3260(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r8
  char v3; // al
  __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  char v7; // [rsp+10h] [rbp-90h]
  _BYTE *v8; // [rsp+18h] [rbp-88h] BYREF
  __int64 v9; // [rsp+20h] [rbp-80h]
  _BYTE v10[120]; // [rsp+28h] [rbp-78h] BYREF

  v2 = (__int64)(a1 + 1);
  v3 = *(_BYTE *)a1;
  v8 = v10;
  v7 = v3;
  v9 = 0x200000000LL;
  if ( *((_DWORD *)a1 + 4) )
  {
    sub_25C2C90((__int64)&v8, a1 + 1);
    v2 = (__int64)(a1 + 1);
  }
  *(_BYTE *)a1 = *(_BYTE *)a2;
  sub_25C2C90(v2, a2 + 1);
  *(_BYTE *)a2 = v7;
  sub_25C2C90((__int64)(a2 + 1), (__int64 *)&v8);
  v4 = (__int64)v8;
  v5 = (unsigned __int64)&v8[32 * (unsigned int)v9];
  if ( v8 != (_BYTE *)v5 )
  {
    do
    {
      v5 -= 32LL;
      if ( *(_DWORD *)(v5 + 24) > 0x40u )
      {
        v6 = *(_QWORD *)(v5 + 16);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
        j_j___libc_free_0_0(*(_QWORD *)v5);
    }
    while ( v4 != v5 );
    v5 = (unsigned __int64)v8;
  }
  if ( (_BYTE *)v5 != v10 )
    _libc_free(v5);
}
