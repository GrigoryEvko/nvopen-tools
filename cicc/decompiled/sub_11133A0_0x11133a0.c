// Function: sub_11133A0
// Address: 0x11133a0
//
__int64 __fastcall sub_11133A0(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD **v9; // rdx
  int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v15; // [rsp+18h] [rbp-38h]

  v9 = *(_QWORD ***)(a3 + 8);
  v10 = *((unsigned __int8 *)v9 + 8);
  if ( (unsigned int)(v10 - 17) > 1 )
  {
    v12 = sub_BCB2A0(*v9);
  }
  else
  {
    BYTE4(v15) = (_BYTE)v10 == 18;
    LODWORD(v15) = *((_DWORD *)v9 + 8);
    v11 = (__int64 *)sub_BCB2A0(*v9);
    v12 = sub_BCE1B0(v11, v15);
  }
  return sub_B523C0(a1, v12, 54, a2, a3, a4, a5, 0, 0, a6);
}
