// Function: sub_F4EE60
// Address: 0xf4ee60
//
unsigned __int64 __fastcall sub_F4EE60(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // rax
  __int64 v11; // r9
  _QWORD *v13; // r15
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int8 v16; // r11
  __int64 v17; // r12

  if ( LOBYTE(qword_4F80F48[8]) )
  {
    v13 = sub_B98A20(a2, a2);
    v14 = sub_B10CD0(a5);
    v15 = sub_22077B0(96);
    v16 = a8;
    v17 = v15;
    if ( v15 )
    {
      sub_B12150(v15, (__int64)v13, a3, a4, v14, 1);
      v16 = a8;
    }
    if ( !a7 )
      BUG();
    return sub_AA8770(*(_QWORD *)(a7 + 16), v17, a7, v16);
  }
  else
  {
    v10 = sub_B10CD0(a5);
    return sub_ADF2E0(a1, a2, a3, a4, v10, v11, a7, (unsigned __int16)a8);
  }
}
