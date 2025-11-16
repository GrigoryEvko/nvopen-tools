// Function: sub_B52500
// Address: 0xb52500
//
__int64 __fastcall sub_B52500(
        int a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdx
  int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // rdx
  int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]
  __int64 v31; // [rsp+38h] [rbp-38h]

  if ( a1 != 53 )
  {
    if ( a7 )
    {
      v12 = sub_BD2C40(72, unk_3F10FD0);
      if ( v12 )
      {
        v13 = *(_QWORD *)(a3 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 > 1 )
        {
          v15 = sub_BCB2A0(*(_QWORD *)v13);
        }
        else
        {
          BYTE4(v30) = *(_BYTE *)(v13 + 8) == 18;
          LODWORD(v30) = *(_DWORD *)(v13 + 32);
          v14 = sub_BCB2A0(*(_QWORD *)v13);
          v15 = sub_BCE1B0(v14, v30);
        }
        sub_B523C0(v12, v15, 54, a2, a3, a4, a5, a7, (unsigned __int16)a8, 0);
      }
      return v12;
    }
    v12 = sub_BD2C40(72, unk_3F10FD0);
    if ( v12 )
    {
      v20 = *(_QWORD **)(a3 + 8);
      v21 = *((unsigned __int8 *)v20 + 8);
      if ( (unsigned int)(v21 - 17) > 1 )
      {
        v23 = sub_BCB2A0(*v20);
      }
      else
      {
        BYTE4(v31) = (_BYTE)v21 == 18;
        LODWORD(v31) = *((_DWORD *)v20 + 8);
        v22 = sub_BCB2A0(*v20);
        v23 = sub_BCE1B0(v22, v31);
      }
      sub_B523C0(v12, v23, 54, a2, a3, a4, a5, 0, 0, 0);
      return v12;
    }
    return 0;
  }
  if ( !a7 )
  {
    v12 = sub_BD2C40(72, unk_3F10FD0);
    if ( v12 )
    {
      v24 = *(_QWORD **)(a3 + 8);
      v25 = *((unsigned __int8 *)v24 + 8);
      if ( (unsigned int)(v25 - 17) > 1 )
      {
        v27 = sub_BCB2A0(*v24);
      }
      else
      {
        BYTE4(v29) = (_BYTE)v25 == 18;
        LODWORD(v29) = *((_DWORD *)v24 + 8);
        v26 = sub_BCB2A0(*v24);
        v27 = sub_BCE1B0(v26, v29);
      }
      sub_B523C0(v12, v27, 53, a2, a3, a4, a5, 0, 0, 0);
      return v12;
    }
    return 0;
  }
  v12 = sub_BD2C40(72, unk_3F10FD0);
  if ( v12 )
  {
    v17 = *(_QWORD *)(a3 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 > 1 )
    {
      v19 = sub_BCB2A0(*(_QWORD *)v17);
    }
    else
    {
      BYTE4(v28) = *(_BYTE *)(v17 + 8) == 18;
      LODWORD(v28) = *(_DWORD *)(v17 + 32);
      v18 = sub_BCB2A0(*(_QWORD *)v17);
      v19 = sub_BCE1B0(v18, v28);
    }
    sub_B523C0(v12, v19, 53, a2, a3, a4, a5, a7, (unsigned __int16)a8, 0);
  }
  return v12;
}
