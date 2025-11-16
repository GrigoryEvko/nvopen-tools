// Function: sub_2A8AFA0
// Address: 0x2a8afa0
//
void __fastcall sub_2A8AFA0(__int64 a1, __int64 *a2)
{
  __int64 *v3; // r13
  bool v5; // al
  __int64 v6; // rdi
  int v7; // r15d
  __int64 v8; // rsi
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  bool v13; // cc
  unsigned __int64 v14; // rdi
  __int64 v15; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v3 = (__int64 *)(a1 + 24);
    while ( a2 != v3 )
    {
      v5 = sub_B445A0(*v3, *(_QWORD *)a1);
      v6 = (__int64)v3;
      v3 += 3;
      if ( v5 )
      {
        v7 = *((_DWORD *)v3 - 2);
        v8 = *(v3 - 3);
        *((_DWORD *)v3 - 2) = 0;
        v9 = *(v3 - 2);
        v10 = 0xAAAAAAAAAAAAAAABLL * ((v6 - a1) >> 3);
        if ( v6 - a1 > 0 )
        {
          for ( *(_QWORD *)v6 = *(_QWORD *)(v6 - 24); ; *(_QWORD *)v6 = v11 )
          {
            *(_QWORD *)(v6 + 8) = *(_QWORD *)(v6 - 16);
            v12 = *(_DWORD *)(v6 - 8);
            *(_DWORD *)(v6 - 8) = 0;
            *(_DWORD *)(v6 + 16) = v12;
            if ( !--v10 )
              break;
            v11 = *(_QWORD *)(v6 - 48);
            v6 -= 24;
          }
        }
        v13 = *(_DWORD *)(a1 + 16) <= 0x40u;
        *(_QWORD *)a1 = v8;
        if ( !v13 )
        {
          v14 = *(_QWORD *)(a1 + 8);
          if ( v14 )
          {
            v15 = v9;
            j_j___libc_free_0_0(v14);
            v9 = v15;
          }
        }
        *(_QWORD *)(a1 + 8) = v9;
        *(_DWORD *)(a1 + 16) = v7;
      }
      else
      {
        sub_2A8AF00(v6);
      }
    }
  }
}
