// Function: sub_15F34F0
// Address: 0x15f34f0
//
void __fastcall sub_15F34F0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  _BYTE *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  _QWORD *v8; // rdi
  __int64 v9; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( *(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0 )
  {
    v2 = sub_1625790(a1, 2);
    v3 = v2;
    if ( v2 )
    {
      if ( *(_DWORD *)(v2 + 8) == 3 )
      {
        v4 = *(_BYTE **)(v2 - 24);
        if ( !*v4 )
        {
          v5 = sub_161E970(v4);
          if ( v6 == 14
            && *(_QWORD *)v5 == 0x775F68636E617262LL
            && *(_DWORD *)(v5 + 8) == 1751607653
            && *(_WORD *)(v5 + 12) == 29556 )
          {
            v7 = *(unsigned int *)(v3 + 8);
            v10[0] = *(_QWORD *)(v3 - 8 * v7);
            v10[1] = *(_QWORD *)(v3 + 8 * (2 - v7));
            v10[2] = *(_QWORD *)(v3 + 8 * (1LL - *(unsigned int *)(v3 + 8)));
            v8 = (_QWORD *)(*(_QWORD *)(v3 + 16) & 0xFFFFFFFFFFFFFFF8LL);
            if ( (*(_QWORD *)(v3 + 16) & 4) != 0 )
              v8 = (_QWORD *)*v8;
            v9 = sub_1627350(v8, v10, 3, 0, 1);
            sub_1625C10(a1, 2, v9);
          }
        }
      }
    }
  }
}
