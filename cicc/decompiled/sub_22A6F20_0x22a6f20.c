// Function: sub_22A6F20
// Address: 0x22a6f20
//
__int64 __fastcall sub_22A6F20(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r14d
  __int64 *v3; // r13
  unsigned __int8 v4; // al
  unsigned int v5; // eax
  unsigned int v6; // eax
  unsigned int v7; // ebx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v11; // ebx
  unsigned __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned int v16; // ebx
  __int16 v17; // [rsp+Dh] [rbp-223h]
  unsigned __int8 v18; // [rsp+Fh] [rbp-221h]
  _QWORD v19[68]; // [rsp+10h] [rbp-220h] BYREF

  v3 = a2;
  sub_AE1D50((__int64)v19);
  v4 = *((_BYTE *)a2 + 10);
  if ( *(_BYTE *)(a1 + 10) < v4 || *(_BYTE *)(a1 + 10) == v4 && *(_DWORD *)(a1 + 12) < *((_DWORD *)a2 + 3) )
    goto LABEL_22;
  if ( sub_22A6B90(a1) && sub_22A6B90((__int64)a2) )
  {
    a2 = v19;
    v2 = sub_22A6C90(a1, (__int64)v19);
    if ( v2 < (unsigned int)sub_22A6C90((__int64)v3, (__int64)v19) )
      goto LABEL_22;
  }
  if ( sub_22A6BA0(a1) && sub_22A6BA0((__int64)v3) )
  {
    v11 = sub_22A6D30(a1);
    if ( v11 < (unsigned int)sub_22A6D30((__int64)v3) )
      goto LABEL_22;
  }
  if ( !sub_22A6B80(a1)
    || (LOBYTE(v5) = sub_22A6B80((__int64)v3), v2 = v5, !(_BYTE)v5)
    || (v12 = sub_22A6C20((__int64)v3),
        v17 = v12,
        v18 = BYTE2(v12),
        v13 = sub_22A6C20(a1),
        (unsigned __int8)v13 >= (unsigned __int8)v17)
    && ((_BYTE)v13 != (_BYTE)v17 || BYTE1(v13) >= HIBYTE(v17) && (BYTE1(v13) != HIBYTE(v17) || BYTE2(v13) >= v18)) )
  {
    if ( !sub_22A6BB0(a1)
      || (LOBYTE(v6) = sub_22A6BB0((__int64)v3), v2 = v6, !(_BYTE)v6)
      || (a2 = v19,
          v14 = sub_22A6D40(v3, (__int64)v19),
          v15 = sub_22A6D40((__int64 *)a1, (__int64)v19),
          (unsigned int)v15 >= (unsigned int)v14)
      && ((_DWORD)v15 != (_DWORD)v14 || HIDWORD(v15) >= HIDWORD(v14)) )
    {
      if ( sub_22A6BF0(a1) && sub_22A6BF0((__int64)v3) )
      {
        v7 = sub_22A6F00(a1);
        if ( v7 < (unsigned int)sub_22A6F00((__int64)v3) )
        {
LABEL_22:
          v2 = 1;
          goto LABEL_23;
        }
      }
      if ( !(unsigned __int8)sub_22A6BC0(a1)
        || (v2 = sub_22A6BC0((__int64)v3), !(_BYTE)v2)
        || (v8 = sub_22A6E00(v3), v9 = sub_22A6E00((__int64 *)a1), (unsigned int)v9 >= (unsigned int)v8)
        && ((_DWORD)v9 != (_DWORD)v8 || HIDWORD(v9) >= HIDWORD(v8)) )
      {
        if ( (unsigned __int8)sub_22A6C00(a1) && (unsigned __int8)sub_22A6C00((__int64)v3) )
        {
          v16 = sub_22A6F10(a1);
          LOBYTE(v2) = v16 < (unsigned int)sub_22A6F10((__int64)v3);
        }
        else
        {
          v2 = 0;
        }
      }
    }
  }
LABEL_23:
  sub_AE4030(v19, (__int64)a2);
  return v2;
}
