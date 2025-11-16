// Function: sub_1111430
// Address: 0x1111430
//
_QWORD *__fastcall sub_1111430(unsigned int *a1, char a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r15d
  __int16 v7; // r15
  _QWORD *v8; // r12
  _QWORD **v9; // rdx
  int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rsi
  _QWORD **v15; // rdx
  int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // [rsp+0h] [rbp-70h]
  __int64 v20; // [rsp+8h] [rbp-68h]
  _BYTE v21[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v6 = *a1;
  if ( (a2 & 4) != 0 )
  {
    v22 = 257;
    v8 = sub_BD2C40(72, unk_3F10FD0);
    if ( v8 )
    {
      v15 = *(_QWORD ***)(a3 + 8);
      v16 = *((unsigned __int8 *)v15 + 8);
      if ( (unsigned int)(v16 - 17) > 1 )
      {
        v18 = sub_BCB2A0(*v15);
      }
      else
      {
        BYTE4(v20) = (_BYTE)v16 == 18;
        LODWORD(v20) = *((_DWORD *)v15 + 8);
        v17 = (__int64 *)sub_BCB2A0(*v15);
        v18 = sub_BCE1B0(v17, v20);
      }
      sub_B523C0((__int64)v8, v18, 53, v6, a3, a4, (__int64)v21, 0, 0, 0);
    }
    *((_BYTE *)v8 + 1) = *((_BYTE *)v8 + 1) & 1 | (2 * ((*((_BYTE *)v8 + 1) >> 1) & 0xFE | ((a2 & 2) != 0)));
  }
  else
  {
    v7 = sub_B52E90(v6);
    v22 = 257;
    v8 = sub_BD2C40(72, unk_3F10FD0);
    if ( v8 )
    {
      v9 = *(_QWORD ***)(a3 + 8);
      v10 = *((unsigned __int8 *)v9 + 8);
      if ( (unsigned int)(v10 - 17) > 1 )
      {
        v12 = sub_BCB2A0(*v9);
      }
      else
      {
        BYTE4(v19) = (_BYTE)v10 == 18;
        LODWORD(v19) = *((_DWORD *)v9 + 8);
        v11 = (__int64 *)sub_BCB2A0(*v9);
        v12 = sub_BCE1B0(v11, v19);
      }
      sub_B523C0((__int64)v8, v12, 53, v7, a3, a4, (__int64)v21, 0, 0, 0);
    }
  }
  return v8;
}
