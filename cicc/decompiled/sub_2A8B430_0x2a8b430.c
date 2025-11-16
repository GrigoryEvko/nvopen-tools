// Function: sub_2A8B430
// Address: 0x2a8b430
//
void __fastcall sub_2A8B430(__int64 a1, __int64 a2)
{
  const void **v3; // r15
  __int64 v4; // r14
  unsigned int v5; // r12d
  _QWORD *v6; // r13
  __int64 v7; // rdi
  bool v8; // al
  __int64 v9; // rdi
  __int64 v10; // r9
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // esi
  bool v15; // cc
  unsigned __int64 v16; // rdi
  __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 24 )
  {
    v3 = (const void **)(a1 + 8);
    v4 = a1 + 48;
    do
    {
      v5 = *(_DWORD *)(v4 - 8);
      v19 = v4;
      v6 = (_QWORD *)(v4 - 24);
      v7 = v4 - 16;
      if ( v5 <= 0x40 )
      {
        if ( *(_QWORD *)(v4 - 16) != *(_QWORD *)(a1 + 8) )
        {
LABEL_6:
          if ( (int)sub_C4C880(v7, (__int64)v3) < 0 )
            goto LABEL_7;
          goto LABEL_17;
        }
      }
      else
      {
        v8 = sub_C43C50(v7, v3);
        v7 = v4 - 16;
        if ( !v8 )
          goto LABEL_6;
      }
      if ( sub_B445A0(*(_QWORD *)(v4 - 24), *(_QWORD *)a1) )
      {
        v5 = *(_DWORD *)(v4 - 8);
LABEL_7:
        *(_DWORD *)(v4 - 8) = 0;
        v9 = *(_QWORD *)(v4 - 24);
        v10 = *(_QWORD *)(v4 - 16);
        v11 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v6 - a1) >> 3);
        if ( (__int64)v6 - a1 > 0 )
        {
          v12 = v4 - 24;
          *v6 = *(_QWORD *)(v4 - 48);
          while ( 1 )
          {
            *(_QWORD *)(v12 + 8) = *(_QWORD *)(v12 - 16);
            v14 = *(_DWORD *)(v12 - 8);
            *(_DWORD *)(v12 - 8) = 0;
            *(_DWORD *)(v12 + 16) = v14;
            if ( !--v11 )
              break;
            v13 = *(_QWORD *)(v12 - 48);
            v12 -= 24;
            *(_QWORD *)v12 = v13;
          }
        }
        v15 = *(_DWORD *)(a1 + 16) <= 0x40u;
        *(_QWORD *)a1 = v9;
        if ( !v15 )
        {
          v16 = *(_QWORD *)(a1 + 8);
          if ( v16 )
          {
            v18 = v10;
            j_j___libc_free_0_0(v16);
            v10 = v18;
          }
        }
        *(_QWORD *)(a1 + 8) = v10;
        *(_DWORD *)(a1 + 16) = v5;
        goto LABEL_15;
      }
LABEL_17:
      sub_2A8B340(v4 - 24);
LABEL_15:
      v4 += 24;
    }
    while ( a2 != v19 );
  }
}
