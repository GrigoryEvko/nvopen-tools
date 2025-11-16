// Function: sub_1113300
// Address: 0x1113300
//
__int64 __fastcall sub_1113300(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD **v9; // rdx
  int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 v14; // [rsp+8h] [rbp-38h]

  v9 = *(_QWORD ***)(a3 + 8);
  v10 = *((unsigned __int8 *)v9 + 8);
  if ( (unsigned int)(v10 - 17) > 1 )
  {
    v12 = sub_BCB2A0(*v9);
  }
  else
  {
    BYTE4(v14) = (_BYTE)v10 == 18;
    LODWORD(v14) = *((_DWORD *)v9 + 8);
    v11 = (__int64 *)sub_BCB2A0(*v9);
    v12 = sub_BCE1B0(v11, v14);
  }
  return sub_B523C0(a1, v12, 53, a2, a3, a4, a5, 0, 0, 0);
}
