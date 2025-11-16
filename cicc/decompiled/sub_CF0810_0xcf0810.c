// Function: sub_CF0810
// Address: 0xcf0810
//
void __fastcall sub_CF0810(unsigned __int64 a1, unsigned __int64 *a2)
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
  __int64 v12; // r12
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // rdi
  _BYTE *v18; // rax
  unsigned __int64 v19[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v19[0] = a1;
    if ( *(_BYTE *)(v2 + 8) == 14 )
    {
      v4 = v19;
      v5 = (__int64)a2;
      sub_CF06F0(a2, v19);
      if ( (_BYTE)v6 )
      {
        v9 = v19[0];
        if ( (*(_DWORD *)(v19[0] + 4) & 0x7FFFFFF) != 0 )
        {
          v10 = 0;
          v11 = 32LL * (*(_DWORD *)(v19[0] + 4) & 0x7FFFFFF);
          do
          {
            if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
              v12 = *(_QWORD *)(v9 - 8);
            else
              v12 = v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
            v5 = *(_QWORD *)(v12 + v10);
            v4 = a2;
            v10 += 32;
            sub_CF0810(v5, a2);
            v9 = v19[0];
          }
          while ( v11 != v10 );
        }
        v13 = sub_CB7210(v5, (__int64)v4, v6, v7, v8);
        sub_A69870(v9, v13, 0);
        v17 = sub_CB7210(v9, (__int64)v13, v14, v15, v16);
        v18 = (_BYTE *)v17[4];
        if ( (_BYTE *)v17[3] == v18 )
        {
          sub_CB6200((__int64)v17, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v18 = 10;
          ++v17[4];
        }
      }
    }
  }
}
