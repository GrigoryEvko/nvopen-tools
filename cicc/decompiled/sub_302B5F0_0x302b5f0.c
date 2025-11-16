// Function: sub_302B5F0
// Address: 0x302b5f0
//
__int64 __fastcall sub_302B5F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // rdi
  _QWORD *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h]

  switch ( *(_BYTE *)a2 )
  {
    case 0:
      sub_302B130(a1, *(_DWORD *)(a2 + 8));
      return 1;
    case 1:
      return 2;
    case 3:
      v10 = *(_QWORD *)(a2 + 24);
      v11 = v10 + 24;
      v12 = *(_BYTE *)(*(_QWORD *)(v10 + 8) + 8LL);
      if ( v12 == 2 )
      {
        v13 = *(_QWORD *)(a1 + 216);
        v14 = 3;
      }
      else if ( v12 > 2u )
      {
        if ( v12 != 3 )
          sub_C64ED0("Unsupported FP type", 1u);
        v13 = *(_QWORD *)(a1 + 216);
        v14 = 4;
      }
      else
      {
        v13 = *(_QWORD *)(a1 + 216);
        if ( v12 )
          v14 = 1;
        else
          v14 = 2;
      }
      sub_30586B0(v14, v11, v13);
      return 5;
    case 4:
      v15 = *(_QWORD **)(a1 + 216);
      v16 = sub_2E309C0(*(_QWORD *)(a2 + 24), a2, a3, a4, a5);
      sub_E808D0(v16, 0, v15, 0);
      return 5;
    case 9:
      if ( **(_BYTE **)(a2 + 24) )
        v18 = *(_QWORD *)(a2 + 24);
      v17 = sub_31DE8D0(a1, a2, 257, a4, a5, a6, v18);
      v7 = *(_QWORD **)(a1 + 216);
      v8 = v17;
      goto LABEL_3;
    case 0xA:
      v6 = sub_31DB510(a1, *(_QWORD *)(a2 + 24));
      v7 = *(_QWORD **)(a1 + 216);
      v8 = v6;
LABEL_3:
      sub_E808D0(v8, 0, v7, 0);
      return 5;
    default:
      BUG();
  }
}
