// Function: sub_1E1A6B0
// Address: 0x1e1a6b0
//
__int64 __fastcall sub_1E1A6B0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 i; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v13; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h]
  __int64 *v15; // [rsp+10h] [rbp-20h] BYREF
  __int16 v16; // [rsp+20h] [rbp-10h]

  v13 = a2;
  LODWORD(a2) = *(_DWORD *)(a1 + 40);
  v14 = a3;
  if ( (_DWORD)a2 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    for ( i = v3 + 40LL * (unsigned int)((_DWORD)a2 - 1); ; i -= 40 )
    {
      if ( *(_BYTE *)i == 14 )
      {
        v5 = *(_QWORD *)(i + 24);
        if ( v5 )
        {
          v6 = *(unsigned int *)(v5 + 8);
          if ( (_DWORD)v6 )
          {
            v7 = *(_QWORD *)(v5 - 8 * v6);
            if ( *(_BYTE *)v7 == 1 )
            {
              v8 = *(_QWORD *)(v7 + 136);
              if ( *(_BYTE *)(v8 + 16) == 13 )
                break;
            }
          }
        }
      }
      if ( v3 == i )
      {
        LODWORD(a2) = 0;
        goto LABEL_12;
      }
    }
    a2 = *(_QWORD **)(v8 + 24);
    if ( *(_DWORD *)(v8 + 32) > 0x40u )
      a2 = (_QWORD *)*a2;
  }
LABEL_12:
  v9 = *(_QWORD *)(a1 + 24);
  if ( !v9 || (v10 = *(_QWORD *)(v9 + 56)) == 0 )
    sub_16BD190((__int64)v13, v14, 1u);
  v11 = **(_QWORD **)(*(_QWORD *)(v10 + 32) + 1688LL);
  v16 = 261;
  v15 = (__int64 *)&v13;
  return sub_1602B40(v11, (int)a2, (__int64)&v15);
}
