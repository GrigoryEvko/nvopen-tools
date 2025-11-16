// Function: sub_1CCC470
// Address: 0x1ccc470
//
void __fastcall sub_1CCC470(unsigned __int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 *v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 i; // r12
  void *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  _QWORD *v16; // rdi
  _BYTE *v17; // rax
  unsigned __int64 v18[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 16) > 0x17u )
  {
    v3 = v18;
    v18[0] = a1;
    v4 = (__int64)a2;
    sub_190CFA0(a2, v18);
    if ( (_BYTE)v5 )
    {
      v8 = v18[0];
      if ( (*(_DWORD *)(v18[0] + 20) & 0xFFFFFFF) != 0 )
      {
        v9 = 0;
        v10 = 24LL * ((*(_DWORD *)(v18[0] + 20) & 0xFFFFFFFu) - 1);
        if ( (*(_BYTE *)(v18[0] + 23) & 0x40) == 0 )
          goto LABEL_8;
LABEL_5:
        for ( i = *(_QWORD *)(v8 - 8); ; i = v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) )
        {
          v4 = *(_QWORD *)(i + v9);
          v3 = a2;
          sub_1CCC470(v4, a2);
          v8 = v18[0];
          if ( v10 == v9 )
            break;
          v9 += 24;
          if ( (*(_BYTE *)(v18[0] + 23) & 0x40) != 0 )
            goto LABEL_5;
LABEL_8:
          ;
        }
      }
      v12 = sub_16E8C20(v4, (__int64)v3, v5, v6, v7);
      sub_155C2B0(v8, (__int64)v12, 0);
      v16 = sub_16E8C20(v8, (__int64)v12, v13, v14, v15);
      v17 = (_BYTE *)v16[3];
      if ( (_BYTE *)v16[2] == v17 )
      {
        sub_16E7EE0((__int64)v16, "\n", 1u);
      }
      else
      {
        *v17 = 10;
        ++v16[3];
      }
    }
  }
}
