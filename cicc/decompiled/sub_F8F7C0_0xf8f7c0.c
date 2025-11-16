// Function: sub_F8F7C0
// Address: 0xf8f7c0
//
__int64 __fastcall sub_F8F7C0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rcx
  unsigned int v13; // r15d
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // rdi
  unsigned __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v18[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !a4
    || (*(_BYTE *)(a3 + 7) & 0x20) != 0 && sub_B91C10(a3, 15)
    || !(unsigned __int8)sub_BC8C50(a3, &v17, v18)
    || !(v18[0] + v17) )
  {
    v8 = *(_QWORD *)(a3 - 32);
    v9 = *(_QWORD *)(a2 - 32);
    if ( v9 == v8 )
    {
LABEL_16:
      *(_BYTE *)a1 = 0;
      *(_DWORD *)(a1 + 4) = 29;
      *(_QWORD *)(a1 + 8) = v8;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
    v10 = *(_QWORD *)(a3 - 64);
    v11 = *(_QWORD *)(a2 - 64);
    if ( v10 != v11 )
    {
      if ( v11 == v8 )
        goto LABEL_25;
      if ( v10 != v9 )
        goto LABEL_8;
      goto LABEL_29;
    }
LABEL_19:
    *(_BYTE *)a1 = 0;
    *(_DWORD *)(a1 + 4) = 28;
    *(_QWORD *)(a1 + 8) = v10;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  v13 = sub_F02DD0(v17, v18[0] + v17);
  v14 = sub_DF95A0(a4);
  v15 = *(_QWORD *)(a3 - 32);
  v9 = *(_QWORD *)(a2 - 32);
  if ( v15 != v9 )
  {
    v16 = *(_QWORD *)(a3 - 64);
    v10 = *(_QWORD *)(a2 - 64);
    if ( v10 != v16 )
    {
      if ( v15 == v10 )
      {
        if ( v13 != -1 && v13 >= v14 )
          goto LABEL_8;
        v11 = *(_QWORD *)(a2 - 64);
LABEL_25:
        *(_BYTE *)a1 = 1;
        *(_DWORD *)(a1 + 4) = 28;
        *(_QWORD *)(a1 + 8) = v11;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      if ( v9 != v16 || v13 != -1 && v14 <= 0x80000000 - v13 )
        goto LABEL_8;
LABEL_29:
      *(_BYTE *)a1 = 1;
      *(_DWORD *)(a1 + 4) = 29;
      *(_QWORD *)(a1 + 8) = v9;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
    if ( v13 != -1 && v14 <= 0x80000000 - v13 )
      goto LABEL_8;
    goto LABEL_19;
  }
  if ( v13 < v14 || v13 == -1 )
  {
    v8 = *(_QWORD *)(a2 - 32);
    goto LABEL_16;
  }
LABEL_8:
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
