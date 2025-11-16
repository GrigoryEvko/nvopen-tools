// Function: sub_1CCC2F0
// Address: 0x1ccc2f0
//
void __fastcall sub_1CCC2F0(unsigned __int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 *v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 i; // r12
  void *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // rdi
  _BYTE *v18; // rax
  unsigned __int64 v19[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 16) > 0x17u )
  {
    v2 = *(_QWORD *)a1;
    v19[0] = a1;
    if ( *(_BYTE *)(v2 + 8) == 15 )
    {
      v4 = v19;
      v5 = (__int64)a2;
      sub_190CFA0(a2, v19);
      if ( (_BYTE)v6 )
      {
        v9 = v19[0];
        if ( (*(_DWORD *)(v19[0] + 20) & 0xFFFFFFF) != 0 )
        {
          v10 = 0;
          v11 = 24LL * ((*(_DWORD *)(v19[0] + 20) & 0xFFFFFFFu) - 1);
          if ( (*(_BYTE *)(v19[0] + 23) & 0x40) == 0 )
            goto LABEL_10;
LABEL_7:
          for ( i = *(_QWORD *)(v9 - 8); ; i = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF) )
          {
            v5 = *(_QWORD *)(i + v10);
            v4 = a2;
            sub_1CCC2F0(v5, a2);
            v9 = v19[0];
            if ( v11 == v10 )
              break;
            v10 += 24;
            if ( (*(_BYTE *)(v19[0] + 23) & 0x40) != 0 )
              goto LABEL_7;
LABEL_10:
            ;
          }
        }
        v13 = sub_16E8C20(v5, (__int64)v4, v6, v7, v8);
        sub_155C2B0(v9, (__int64)v13, 0);
        v17 = sub_16E8C20(v9, (__int64)v13, v14, v15, v16);
        v18 = (_BYTE *)v17[3];
        if ( (_BYTE *)v17[2] == v18 )
        {
          sub_16E7EE0((__int64)v17, "\n", 1u);
        }
        else
        {
          *v18 = 10;
          ++v17[3];
        }
      }
    }
  }
}
