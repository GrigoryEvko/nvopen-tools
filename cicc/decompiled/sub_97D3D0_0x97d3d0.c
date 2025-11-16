// Function: sub_97D3D0
// Address: 0x97d3d0
//
__int64 __fastcall sub_97D3D0(__int64 a1, __int64 a2, _BYTE *a3, _BYTE **a4, __int64 a5, int a6)
{
  int v8; // r13d
  _BYTE **v11; // rax
  _BYTE **v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  _BYTE **v15; // rcx
  _BYTE *v16; // rax
  __int64 result; // rax
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+20h] [rbp-50h]
  unsigned int v23; // [rsp+28h] [rbp-48h]
  char v24; // [rsp+30h] [rbp-40h]

  v8 = a2;
  if ( (unsigned __int8)sub_BCEA30(a2) || *a3 > 0x15u )
    return 0;
  v11 = a4;
  v12 = &a4[a5];
  v13 = (8 * a5) >> 5;
  v14 = (8 * a5) >> 3;
  if ( v13 > 0 )
  {
    v15 = &a4[4 * v13];
    while ( **v11 <= 0x15u )
    {
      if ( *v11[1] > 0x15u )
      {
        if ( v12 == v11 + 1 )
          goto LABEL_11;
        return 0;
      }
      if ( *v11[2] > 0x15u )
      {
        if ( v12 == v11 + 2 )
          goto LABEL_11;
        return 0;
      }
      if ( *v11[3] > 0x15u )
      {
        if ( v12 == v11 + 3 )
          goto LABEL_11;
        return 0;
      }
      v11 += 4;
      if ( v11 == v15 )
      {
        v14 = v12 - v11;
        goto LABEL_21;
      }
    }
LABEL_10:
    if ( v12 == v11 )
      goto LABEL_11;
    return 0;
  }
LABEL_21:
  if ( v14 != 2 )
  {
    if ( v14 != 3 )
    {
      if ( v14 != 1 )
        goto LABEL_11;
      goto LABEL_24;
    }
    if ( **v11 > 0x15u )
      goto LABEL_10;
    ++v11;
  }
  if ( **v11 > 0x15u )
    goto LABEL_10;
  ++v11;
LABEL_24:
  if ( **v11 > 0x15u )
    goto LABEL_10;
LABEL_11:
  v24 = 0;
  v16 = (_BYTE *)sub_AD9FD0(v8, (_DWORD)a3, (_DWORD)a4, a5, a6, (unsigned int)&v20, 0);
  result = sub_97B670(v16, *(_QWORD *)(a1 + 8), 0);
  if ( v24 )
  {
    v24 = 0;
    if ( v23 > 0x40 && v22 )
    {
      v18 = result;
      j_j___libc_free_0_0(v22);
      result = v18;
    }
    if ( v21 > 0x40 )
    {
      if ( v20 )
      {
        v19 = result;
        j_j___libc_free_0_0(v20);
        return v19;
      }
    }
  }
  return result;
}
