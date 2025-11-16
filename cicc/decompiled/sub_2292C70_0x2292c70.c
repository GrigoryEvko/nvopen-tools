// Function: sub_2292C70
// Address: 0x2292c70
//
void __fastcall sub_2292C70(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // r13
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  char v12; // al
  __int64 *v13; // rax
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // rdi

  LODWORD(v3) = *a3;
  if ( *a3 != 4 )
  {
    switch ( (_DWORD)v3 )
    {
      case 2:
        *(_BYTE *)a2 &= ~8u;
        v13 = sub_228CE10((__int64)a3);
        *(_QWORD *)(a2 + 8) = v13;
        v14 = sub_DBE090(*(_QWORD *)(a1 + 8), (__int64)v13);
        v15 = *(_QWORD *)(a2 + 8);
        v16 = *(_QWORD *)(a1 + 8);
        if ( v14 )
          LOBYTE(v3) = 0;
        if ( !(unsigned __int8)sub_DBEC80(v16, v15) )
          LOBYTE(v3) = v3 | 1;
        v12 = sub_DBED40(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8));
        break;
      case 3:
        *(_BYTE *)a2 &= ~8u;
        *(_QWORD *)(a2 + 8) = 0;
        return;
      case 1:
        *(_BYTE *)a2 &= ~8u;
        *(_QWORD *)(a2 + 8) = 0;
        v3 = sub_228CDC0((__int64)a3);
        v7 = sub_228CDD0((__int64)a3);
        LOBYTE(v3) = 2 * (sub_228DFC0(a1, 0x21u, v7, v3) == 0);
        v8 = sub_228CDC0((__int64)a3);
        v9 = sub_228CDD0((__int64)a3);
        if ( !sub_228DFC0(a1, 0x29u, v9, v8) )
          LOBYTE(v3) = v3 | 1;
        v10 = sub_228CDC0((__int64)a3);
        v11 = sub_228CDD0((__int64)a3);
        v12 = sub_228DFC0(a1, 0x27u, v11, v10);
        break;
      default:
        BUG();
    }
    if ( !v12 )
      LOBYTE(v3) = v3 | 4;
    *(_BYTE *)a2 = *(_BYTE *)a2 & 7 & v3 | *(_BYTE *)a2 & 0xF8;
  }
}
