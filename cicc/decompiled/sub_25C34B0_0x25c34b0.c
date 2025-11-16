// Function: sub_25C34B0
// Address: 0x25c34b0
//
void __fastcall sub_25C34B0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r12
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // r14
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rdi
  __int64 v14; // [rsp+10h] [rbp-90h]
  char v15; // [rsp+18h] [rbp-88h]
  _BYTE *v16; // [rsp+20h] [rbp-80h] BYREF
  __int64 v17; // [rsp+28h] [rbp-78h]
  _BYTE v18[112]; // [rsp+30h] [rbp-70h] BYREF

  if ( (__int64 *)a1 != a2 )
  {
    v2 = (__int64 *)(a1 + 96);
    while ( a2 != v2 )
    {
      v4 = *v2;
      v5 = (__int64)v2;
      v2 += 12;
      if ( sub_B445A0(v4, *(_QWORD *)a1) )
      {
        v14 = *(v2 - 12);
        v15 = *((_BYTE *)v2 - 88);
        v16 = v18;
        v17 = 0x200000000LL;
        if ( *((_DWORD *)v2 - 18) )
          sub_25C2C90((__int64)&v16, v2 - 10);
        v6 = v2 - 10;
        v7 = v5 - a1;
        v8 = 0xAAAAAAAAAAAAAAABLL * ((v5 - a1) >> 5);
        if ( v7 > 0 )
        {
          do
          {
            v9 = *(v6 - 14);
            v10 = (__int64)v6;
            v6 -= 12;
            v6[10] = v9;
            *((_BYTE *)v6 + 88) = *((_BYTE *)v6 - 8);
            sub_25C2C90(v10, v6);
            --v8;
          }
          while ( v8 );
        }
        *(_QWORD *)a1 = v14;
        *(_BYTE *)(a1 + 8) = v15;
        sub_25C2C90(a1 + 16, (__int64 *)&v16);
        v11 = (__int64)v16;
        v12 = (unsigned __int64)&v16[32 * (unsigned int)v17];
        if ( v16 != (_BYTE *)v12 )
        {
          do
          {
            v12 -= 32LL;
            if ( *(_DWORD *)(v12 + 24) > 0x40u )
            {
              v13 = *(_QWORD *)(v12 + 16);
              if ( v13 )
                j_j___libc_free_0_0(v13);
            }
            if ( *(_DWORD *)(v12 + 8) > 0x40u && *(_QWORD *)v12 )
              j_j___libc_free_0_0(*(_QWORD *)v12);
          }
          while ( v11 != v12 );
          v12 = (unsigned __int64)v16;
        }
        if ( (_BYTE *)v12 != v18 )
          _libc_free(v12);
      }
      else
      {
        sub_25C3380(v5);
      }
    }
  }
}
