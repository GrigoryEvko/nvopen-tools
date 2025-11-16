// Function: sub_39C9540
// Address: 0x39c9540
//
void __fastcall sub_39C9540(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v6; // rax
  unsigned __int8 *v7; // r13
  char v8; // al
  __int64 v9; // rdx
  _BYTE *v10; // rsi
  __int64 v11; // rdx

  v4 = *(unsigned int *)(a2 + 8);
  v6 = *(_QWORD *)(a2 + 8 * (6 - v4));
  if ( v6 )
    v7 = *(unsigned __int8 **)(v6 + 8 * (1LL - *(unsigned int *)(v6 + 8)));
  else
    v7 = *(unsigned __int8 **)(a2 + 8 * (1 - v4));
  v8 = sub_39C84F0(a1);
  sub_39A70C0((__int64)a1, a2, a3, v8);
  v9 = *(unsigned int *)(a2 + 8);
  v10 = *(_BYTE **)(a2 + 8 * (2 - v9));
  if ( v10 )
    v10 = (_BYTE *)sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v9)));
  else
    v11 = 0;
  sub_39C8580((__int64)a1, v10, v11, a3, v7);
}
