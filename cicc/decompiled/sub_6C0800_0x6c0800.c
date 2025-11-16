// Function: sub_6C0800
// Address: 0x6c0800
//
__int64 __fastcall sub_6C0800(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  char i; // dl
  __m128i *v6; // [rsp+0h] [rbp-60h]
  unsigned __int8 v7; // [rsp+1Fh] [rbp-41h] BYREF
  __int64 v8; // [rsp+20h] [rbp-40h] BYREF
  int v9; // [rsp+28h] [rbp-38h] BYREF
  __int16 v10; // [rsp+2Ch] [rbp-34h]
  int v11; // [rsp+30h] [rbp-30h] BYREF
  _QWORD v12[5]; // [rsp+38h] [rbp-28h] BYREF

  v7 = 0;
  if ( (unsigned int)sub_6AD110(3, (__int64 *)a1, &v9, &v8, &v11, v12, v6) )
  {
    sub_6BFEC0(v8, (__int64 *)a2, &v9, &v11, &v7);
    if ( a1 )
    {
      if ( !*(_BYTE *)(a2 + 16) )
        goto LABEL_9;
      v4 = *(_QWORD *)a2;
      for ( i = *(_BYTE *)(*(_QWORD *)a2 + 140LL); i == 12; i = *(_BYTE *)(v4 + 140) )
        v4 = *(_QWORD *)(v4 + 160);
      if ( !i )
LABEL_9:
        *(_BYTE *)(a1 + 56) = 1;
    }
  }
  else
  {
    sub_6E6840(a2);
  }
  *(_DWORD *)(a2 + 68) = v9;
  *(_WORD *)(a2 + 72) = v10;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  v2 = v12[0];
  *(_QWORD *)(a2 + 76) = v12[0];
  unk_4F061D8 = v2;
  sub_6E3280(a2, &v9);
  return sub_6E26D0(v7, a2);
}
