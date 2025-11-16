// Function: sub_921B80
// Address: 0x921b80
//
__int64 __fastcall sub_921B80(__int64 a1, __int64 a2, __int64 a3, unsigned __int16 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned int *v9; // rbx
  unsigned int *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int16 v13; // dx
  __int64 v15; // r13
  __int64 v16; // rax
  int v17; // [rsp+8h] [rbp-78h]
  unsigned __int8 v18; // [rsp+13h] [rbp-6Dh]
  int v19; // [rsp+14h] [rbp-6Ch]
  char v21; // [rsp+20h] [rbp-60h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  if ( a5 )
  {
    v17 = sub_92F410(a1, a5);
    v6 = sub_AA4E30(*(_QWORD *)(a1 + 96));
    v18 = sub_AE5260(v6, a2);
    v19 = *(_DWORD *)(v6 + 4);
    v22 = 257;
    v7 = sub_BD2C40(80, unk_3F10A14);
    v8 = v7;
    if ( v7 )
      sub_B4CCA0(v7, a2, v19, v17, v18, (unsigned int)&v21, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      v8,
      a3,
      *(_QWORD *)(a1 + 104),
      *(_QWORD *)(a1 + 112));
    v9 = *(unsigned int **)(a1 + 48);
    v10 = &v9[4 * *(unsigned int *)(a1 + 56)];
    while ( v10 != v9 )
    {
      v11 = *((_QWORD *)v9 + 1);
      v12 = *v9;
      v9 += 4;
      sub_B99FD0(v8, v12, v11);
    }
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 456) + 24LL;
    v16 = sub_BD2C40(80, unk_3F10A14);
    v8 = v16;
    if ( v16 )
      sub_B4CE50(v16, a2, 0, a3, v15, 0);
  }
  v13 = 0;
  if ( HIBYTE(a4) )
    v13 = (unsigned __int8)a4;
  *(_WORD *)(v8 + 2) = v13 | *(_WORD *)(v8 + 2) & 0xFFC0;
  return v8;
}
