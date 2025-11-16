// Function: sub_31D5550
// Address: 0x31d5550
//
__int64 __fastcall sub_31D5550(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 (*v6)(); // rax
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r9
  _QWORD *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax

  v6 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a2 + 232) + 16LL) + 144LL);
  if ( v6 == sub_2C8F680 )
    BUG();
  v9 = (_QWORD *)v6();
  v10 = *(_QWORD *)(a2 + 232);
  v11 = *v9;
  v12 = v9;
  v13 = *(_QWORD *)(a2 + 240);
  v14 = *(_QWORD *)(v13 + 2480);
  v15 = v13 + 8;
  if ( !v14 )
    v14 = v15;
  v16 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, __int64))(v11 + 1944))(v12, v10, a3, v14) + 16);
  *(_QWORD *)(a1 + 8) = a5;
  *(_WORD *)a1 = 4;
  *(_QWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 16) = 0;
  return a1;
}
