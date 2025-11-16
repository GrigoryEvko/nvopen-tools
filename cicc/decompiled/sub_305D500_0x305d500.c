// Function: sub_305D500
// Address: 0x305d500
//
__int64 __fastcall sub_305D500(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v3; // rsi
  __int64 v4; // rbx
  __int64 (__fastcall *v5)(__int64); // rax
  __int64 v6; // rdi
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  __int16 v13; // ax
  _QWORD *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rsi

  result = sub_2FF0570(a1);
  if ( (_DWORD)result && !(_BYTE)qword_502C2C8 )
  {
    v3 = (_QWORD *)sub_2D5CDB0();
    sub_2FF0E80(a1, v3, 0);
    v4 = *(_QWORD *)(a1 + 256);
    if ( !(_BYTE)qword_502C1E8 )
    {
      v5 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v4 + 1288) + 144LL);
      if ( v5 == sub_3020010 )
        v6 = v4 + 2248;
      else
        v6 = v5(v4 + 1288);
      v7 = (_QWORD *)sub_2C94810(v6);
      sub_2FF0E80(a1, v7, 0);
      v8 = (_QWORD *)sub_271ED00();
      sub_2FF0E80(a1, v8, 0);
    }
    v9 = (_QWORD *)sub_2CFE160(1);
    sub_2FF0E80(a1, v9, 0);
    if ( (_BYTE)qword_502C3A8 )
    {
      v15 = (_QWORD *)sub_2977E10(1);
      sub_2FF0E80(a1, v15, 0);
    }
    v10 = (_QWORD *)sub_271ED00();
    sub_2FF0E80(a1, v10, 0);
    if ( !(_BYTE)qword_502C108 )
    {
      LOBYTE(v13) = byte_502BD88 ^ 1;
      HIBYTE(v13) = byte_502BCA8 ^ 1;
      v14 = (_QWORD *)sub_2D1A100(v13);
      sub_2FF0E80(a1, v14, 0);
    }
    if ( !(_BYTE)qword_502BBC8 )
    {
      v12 = (_QWORD *)sub_2D1A3C0();
      sub_2FF0E80(a1, v12, 0);
    }
    if ( (_BYTE)qword_502C028 )
    {
      if ( *(_DWORD *)(v4 + 1632) > 0x1Fu )
      {
        v16 = (_QWORD *)sub_3085950();
        sub_2FF0E80(a1, v16, 0);
      }
    }
    v11 = (_QWORD *)sub_31CE1A0();
    return sub_2FF0E80(a1, v11, 0);
  }
  return result;
}
