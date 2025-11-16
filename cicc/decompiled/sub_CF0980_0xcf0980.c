// Function: sub_CF0980
// Address: 0xcf0980
//
void __fastcall sub_CF0980(_BYTE *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // r12
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  _QWORD *v16; // rdi
  _BYTE *v17; // rax
  unsigned __int64 v18[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( *a1 > 0x1Cu )
  {
    v3 = v18;
    v18[0] = (unsigned __int64)a1;
    v4 = (__int64)a2;
    sub_CF06F0(a2, v18);
    if ( (_BYTE)v5 )
    {
      v8 = v18[0];
      if ( (*(_DWORD *)(v18[0] + 4) & 0x7FFFFFF) != 0 )
      {
        v9 = 0;
        v10 = 32LL * (*(_DWORD *)(v18[0] + 4) & 0x7FFFFFF);
        do
        {
          if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
            v11 = *(_QWORD *)(v8 - 8);
          else
            v11 = v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
          v4 = *(_QWORD *)(v11 + v9);
          v3 = a2;
          v9 += 32;
          sub_CF0980(v4, a2);
          v8 = v18[0];
        }
        while ( v10 != v9 );
      }
      v12 = sub_CB7210(v4, (__int64)v3, v5, v6, v7);
      sub_A69870(v8, v12, 0);
      v16 = sub_CB7210(v8, (__int64)v12, v13, v14, v15);
      v17 = (_BYTE *)v16[4];
      if ( (_BYTE *)v16[3] == v17 )
      {
        sub_CB6200((__int64)v16, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v17 = 10;
        ++v16[4];
      }
    }
  }
}
