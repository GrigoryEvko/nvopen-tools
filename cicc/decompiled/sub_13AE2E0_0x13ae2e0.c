// Function: sub_13AE2E0
// Address: 0x13ae2e0
//
void __fastcall sub_13AE2E0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // r13
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rax
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // rdi

  LODWORD(v3) = *a3;
  if ( *a3 != 4 )
  {
    *(_BYTE *)a2 &= ~8u;
    if ( (_DWORD)v3 == 2 )
    {
      v13 = sub_13A62A0((__int64)a3);
      *(_QWORD *)(a2 + 8) = v13;
      v14 = sub_1477CE0(*(_QWORD *)(a1 + 8), v13);
      v15 = *(_QWORD *)(a2 + 8);
      v16 = *(_QWORD *)(a1 + 8);
      if ( v14 )
        LOBYTE(v3) = 0;
      if ( !(unsigned __int8)sub_1477A90(v16, v15) )
        LOBYTE(v3) = v3 | 1;
      v12 = sub_1477BC0(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8));
    }
    else
    {
      *(_QWORD *)(a2 + 8) = 0;
      if ( (_DWORD)v3 == 3 )
        return;
      v3 = sub_13A6250((__int64)a3);
      v7 = sub_13A6260((__int64)a3);
      LOBYTE(v3) = 2 * ((unsigned __int8)sub_13A7760(a1, 33, v7, v3) == 0);
      v8 = sub_13A6250((__int64)a3);
      v9 = sub_13A6260((__int64)a3);
      if ( !(unsigned __int8)sub_13A7760(a1, 41, v9, v8) )
        LOBYTE(v3) = v3 | 1;
      v10 = sub_13A6250((__int64)a3);
      v11 = sub_13A6260((__int64)a3);
      v12 = sub_13A7760(a1, 39, v11, v10);
    }
    if ( !v12 )
      LOBYTE(v3) = v3 | 4;
    *(_BYTE *)a2 = *(_BYTE *)a2 & 7 & v3 | *(_BYTE *)a2 & 0xF8;
  }
}
