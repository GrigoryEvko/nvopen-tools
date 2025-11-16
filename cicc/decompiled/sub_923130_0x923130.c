// Function: sub_923130
// Address: 0x923130
//
__int64 __fastcall sub_923130(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, char a5)
{
  int v5; // r10d
  __int64 v6; // rax
  unsigned __int8 v7; // al
  int v8; // ebx
  __int64 v9; // rax
  int v10; // r9d
  __int64 v11; // r13
  __int64 result; // rax
  unsigned int *v13; // rbx
  unsigned int *i; // r12
  __int64 v15; // rdx
  __int64 v16; // rsi
  bool v17; // al
  int v18; // [rsp+0h] [rbp-70h]
  int v19; // [rsp+0h] [rbp-70h]
  int v20; // [rsp+0h] [rbp-70h]
  int v21; // [rsp+8h] [rbp-68h]
  int v22; // [rsp+8h] [rbp-68h]
  int v23; // [rsp+8h] [rbp-68h]
  _BYTE v24[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v25; // [rsp+30h] [rbp-40h]

  v5 = 1;
  if ( !a5 )
  {
    v5 = unk_4D0463C;
    if ( unk_4D0463C )
    {
      v20 = a4;
      v23 = a3;
      v17 = sub_90AA40(*(_QWORD *)(a1 + 32), a3);
      LODWORD(a4) = v20;
      LODWORD(a3) = v23;
      v5 = v17;
    }
  }
  if ( (_DWORD)a4 )
  {
    _BitScanReverse64((unsigned __int64 *)&a4, (unsigned int)a4);
    v8 = (unsigned __int8)(63 - (a4 ^ 0x3F));
  }
  else
  {
    v18 = a3;
    v21 = v5;
    v6 = sub_AA4E30(*(_QWORD *)(a1 + 96));
    v7 = sub_AE5020(v6, *(_QWORD *)(a2 + 8));
    v5 = v21;
    LODWORD(a3) = v18;
    v8 = v7;
  }
  v19 = a3;
  v22 = v5;
  v25 = 257;
  v9 = sub_BD2C40(80, unk_3F10A10);
  v11 = v9;
  if ( v9 )
    sub_B4D3C0(v9, a2, v19, v22, v8, v10, 0, 0);
  result = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
             *(_QWORD *)(a1 + 136),
             v11,
             v24,
             *(_QWORD *)(a1 + 104),
             *(_QWORD *)(a1 + 112));
  v13 = *(unsigned int **)(a1 + 48);
  for ( i = &v13[4 * *(unsigned int *)(a1 + 56)]; i != v13; result = sub_B99FD0(v11, v16, v15) )
  {
    v15 = *((_QWORD *)v13 + 1);
    v16 = *v13;
    v13 += 4;
  }
  return result;
}
