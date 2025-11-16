// Function: sub_30F0800
// Address: 0x30f0800
//
void __fastcall sub_30F0800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // r13
  unsigned int v9; // r15d
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  const char *v12; // [rsp+0h] [rbp-70h] BYREF
  _BYTE *v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h]
  int v15; // [rsp+18h] [rbp-58h]
  char v16; // [rsp+1Ch] [rbp-54h]
  _BYTE v17[80]; // [rsp+20h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a2 - 32);
  v13 = v17;
  v12 = 0;
  v14 = 4;
  v15 = 0;
  v16 = 1;
  v8 = sub_30EFD90((__int64 *)a1, v7, 0, (__int64)&v12, a5, a6);
  if ( !v16 )
    _libc_free((unsigned __int64)v13);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      if ( (unsigned __int64)(*(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8) > *(_QWORD *)(v8 + 24) )
        return;
LABEL_9:
      v17[1] = 1;
      v12 = "Undefined result: Shift count out of range";
      v17[0] = 3;
      sub_CA0E80((__int64)&v12, a1 + 88);
      v10 = *(_BYTE **)(a1 + 120);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 112) )
      {
        sub_CB5D20(a1 + 88, 10);
      }
      else
      {
        *(_QWORD *)(a1 + 120) = v10 + 1;
        *v10 = 10;
      }
      if ( *(_BYTE *)a2 <= 0x1Cu )
      {
        sub_A5BF40((unsigned __int8 *)a2, a1 + 88, 1, *(_QWORD *)a1);
        v11 = *(_BYTE **)(a1 + 120);
        if ( (unsigned __int64)v11 < *(_QWORD *)(a1 + 112) )
          goto LABEL_13;
      }
      else
      {
        sub_A69870(a2, (_BYTE *)(a1 + 88), 0);
        v11 = *(_BYTE **)(a1 + 120);
        if ( (unsigned __int64)v11 < *(_QWORD *)(a1 + 112) )
        {
LABEL_13:
          *(_QWORD *)(a1 + 120) = v11 + 1;
          *v11 = 10;
          return;
        }
      }
      sub_CB5D20(a1 + 88, 10);
      return;
    }
    if ( v9 - (unsigned int)sub_C444A0(v8 + 24) > 0x40
      || (unsigned __int64)(*(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8) <= **(_QWORD **)(v8 + 24) )
    {
      goto LABEL_9;
    }
  }
}
