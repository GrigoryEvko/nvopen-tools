// Function: sub_7F2990
// Address: 0x7f2990
//
char __fastcall sub_7F2990(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rbx
  __int64 v3; // rax
  __m128i *v4; // r13
  _QWORD *v5; // rax
  _BYTE *v6; // rax
  __int64 *v7; // rax
  _BYTE *v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // r12

  v1 = *(_QWORD *)(a1 + 72);
  v2 = *(_QWORD **)(v1 + 136);
  if ( v2 )
  {
    while ( 1 )
    {
      v10 = v2[5];
      if ( *(_BYTE *)(v10 + 24) != 1 || (*(_BYTE *)(v10 + 58) & 1) == 0 )
        goto LABEL_4;
      if ( *(_BYTE *)(v10 + 56) == 73 )
      {
        v3 = sub_72D2E0(*(_QWORD **)v10);
        v4 = sub_7E7CA0(v3);
        v5 = sub_73B8B0(*(const __m128i **)(v10 + 72), 0);
        v6 = sub_73E1B0((__int64)v5, 0);
        v7 = (__int64 *)sub_7E2BE0((__int64)v4, (__int64)v6);
        v8 = sub_73DF90(v10, v7);
        v9 = sub_73DCD0(v8);
        v2[5] = v9;
        v10 = (__int64)v9;
LABEL_4:
        LOBYTE(v1) = sub_7F2600(v10, 0);
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          return v1;
      }
      else
      {
        LOBYTE(v1) = sub_6851C0(0x7BFu, (_DWORD *)(v10 + 28));
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          return v1;
      }
    }
  }
  return v1;
}
