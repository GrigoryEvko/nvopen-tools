// Function: sub_7E26B0
// Address: 0x7e26b0
//
void __fastcall sub_7E26B0(const __m128i *a1, __int64 a2, __int64 a3, char a4, int *a5)
{
  const __m128i *v5; // r9
  __int64 v7; // r13
  __int64 v8; // r12
  _QWORD *v10; // r15
  __int64 v11; // rax
  _BYTE *v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // rax
  _BYTE *v15; // r15
  _QWORD *v16; // rax
  void *v17; // rax
  _QWORD *v18; // r15
  __int64 v19; // rax
  _BYTE *v20; // r15
  __int64 v21; // rax
  void *v22; // r13
  _QWORD *v23; // rax

  v5 = a1;
  v7 = a2;
  v8 = a3;
  if ( !a4 )
  {
    if ( !a3 )
      return;
LABEL_5:
    v18 = sub_73B8B0(v5, 0);
    v19 = sub_7E1C30();
    v20 = sub_73E130(v18, v19);
    *((_QWORD *)v20 + 2) = sub_73A830(v7, byte_4F06A51[0]);
    v21 = sub_7E1C30();
    v22 = sub_73DBF0(0x32u, v21, (__int64)v20);
    v23 = sub_73A830(v8, byte_4F06A51[0]);
    sub_7FA9D0(v22, v23, a5);
    return;
  }
  v8 = a3 - 1;
  v10 = sub_73B8B0(a1, 0);
  v11 = sub_7E1C30();
  v7 = a2 + 1;
  v12 = sub_73E130(v10, v11);
  *((_QWORD *)v12 + 2) = sub_73A830(a2, byte_4F06A51[0]);
  v13 = sub_7E1C30();
  v14 = sub_73DBF0(0x32u, v13, (__int64)v12);
  v15 = sub_73DCD0(v14);
  *((_QWORD *)v15 + 2) = sub_73A830((1 << a4) - 1, byte_4F068B0[0]);
  v16 = sub_72BA30(byte_4F068B0[0]);
  v17 = sub_73DBF0(0x51u, (__int64)v16, (__int64)v15);
  sub_7E25D0((__int64)v17, a5);
  v5 = a1;
  if ( v8 )
    goto LABEL_5;
}
