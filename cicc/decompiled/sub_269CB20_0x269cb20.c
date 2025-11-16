// Function: sub_269CB20
// Address: 0x269cb20
//
__int64 __fastcall sub_269CB20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rdi
  unsigned __int8 v9; // al
  unsigned int v10; // [rsp+Ch] [rbp-F4h]
  unsigned int v11; // [rsp+1Ch] [rbp-E4h] BYREF
  __int64 v12[4]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v13; // [rsp+40h] [rbp-C0h] BYREF
  char *v14; // [rsp+48h] [rbp-B8h]
  __int64 v15; // [rsp+50h] [rbp-B0h]
  int v16; // [rsp+58h] [rbp-A8h]
  char v17; // [rsp+5Ch] [rbp-A4h]
  char v18; // [rsp+60h] [rbp-A0h] BYREF

  result = 1;
  v11 = 1;
  if ( (_BYTE)qword_4FF4B28 )
    return result;
  v4 = *(__int64 **)(a1 + 320);
  v12[2] = (__int64)&v13;
  v12[3] = (__int64)&v11;
  v5 = *(unsigned int *)(a1 + 328);
  v13 = 0;
  v6 = &v4[v5];
  v14 = &v18;
  v15 = 16;
  v16 = 0;
  v17 = 1;
  v12[0] = a1;
  v12[1] = a2;
  while ( v6 != v4 )
  {
    v7 = *v4++;
    sub_269C5D0(v12, v7);
  }
  v8 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
  {
    v8 = *(_QWORD *)(v8 + 24);
    v9 = *(_BYTE *)v8;
    if ( !*(_BYTE *)v8 )
      goto LABEL_9;
  }
  else
  {
    v9 = *(_BYTE *)v8;
    if ( !*(_BYTE *)v8 )
      goto LABEL_9;
  }
  if ( v9 == 22 )
  {
    if ( !(unsigned __int8)sub_26747F0(*(_QWORD *)(v8 + 24)) )
    {
LABEL_10:
      result = v11;
      if ( v17 )
        return result;
LABEL_14:
      v10 = result;
      _libc_free((unsigned __int64)v14);
      return v10;
    }
    goto LABEL_15;
  }
  if ( v9 <= 0x1Cu )
    v8 = 0;
  else
    v8 = sub_B43CB0(v8);
LABEL_9:
  if ( !(unsigned __int8)sub_26747F0(v8) )
    goto LABEL_10;
LABEL_15:
  sub_269C5D0(v12, 0);
  result = v11;
  if ( !v17 )
    goto LABEL_14;
  return result;
}
