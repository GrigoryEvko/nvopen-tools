// Function: sub_6ECC10
// Address: 0x6ecc10
//
void __fastcall sub_6ECC10(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  unsigned int v5; // r13d
  _DWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // [rsp-20h] [rbp-20h] BYREF
  int v10; // [rsp-1Ch] [rbp-1Ch] BYREF

  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    v3 = *(_QWORD *)(a1 + 144);
    if ( *(_BYTE *)(v3 + 24) == 3 )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
      {
        v4 = *(_QWORD *)(v3 + 56);
        if ( (*(_DWORD *)(v4 + 168) & 0x501000) == 0x100000 )
        {
          if ( **(_QWORD **)(v4 + 216) )
          {
            if ( *(char *)(v4 + 170) >= 0 )
            {
              v7 = sub_892240(*(_QWORD *)v4, a2);
              v8 = *(_QWORD *)(v7 + 16);
              if ( v8 )
              {
                if ( (*(_BYTE *)(*(_QWORD *)(v7 + 32) + 81LL) & 2) != 0 && (*(_BYTE *)(v8 + 28) & 1) == 0 )
                  sub_8AA320(v7, 0, 1);
              }
            }
          }
        }
      }
    }
    else
    {
      v10 = 0;
      sub_6DFA90(v3, &v10, &v9);
      if ( v10 )
      {
        v5 = v9;
        v6 = (_DWORD *)(a1 + 68);
        if ( sub_6E53E0(5, v9, v6) )
          sub_684B30(v5, v6);
      }
    }
  }
}
