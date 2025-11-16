// Function: sub_F209F0
// Address: 0xf209f0
//
_BYTE *__fastcall sub_F209F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int16 v4; // ax
  _BYTE *v5; // r14
  __int64 v6; // rax
  _BYTE **v7; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int8 v10; // bl
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-B8h]
  _BYTE *v15; // [rsp+18h] [rbp-A8h] BYREF
  _BYTE v16[32]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v17; // [rsp+40h] [rbp-80h]
  __int64 v18; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v19[4]; // [rsp+58h] [rbp-68h] BYREF
  __int16 v20; // [rsp+78h] [rbp-48h]
  _QWORD v21[8]; // [rsp+80h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a1 + 32);
  v19[0] = 0;
  v19[1] = 0;
  v3 = *(_QWORD *)(v2 + 48);
  v18 = v2;
  v19[2] = v3;
  if ( v3 != -4096 && v3 != 0 && v3 != -8192 )
    sub_BD73F0((__int64)v19);
  v4 = *(_WORD *)(v2 + 64);
  v19[3] = *(_QWORD *)(v2 + 56);
  v20 = v4;
  sub_B33910(v21, (__int64 *)v2);
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v15 = (_BYTE *)sub_1028510(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 88), a2, 0);
    v5 = v15;
  }
  else
  {
    sub_D5F1F0(*(_QWORD *)(a1 + 32), a2);
    v15 = (_BYTE *)sub_1028510(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 88), a2, 0);
    v5 = v15;
    v6 = *(_QWORD *)(a2 + 16);
    if ( !v6 || *(_QWORD *)(v6 + 8) )
    {
      v7 = (_BYTE **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      if ( (_BYTE **)a2 != v7 )
      {
        while ( **v7 == 17 )
        {
          v7 += 4;
          if ( (_BYTE **)a2 == v7 )
            goto LABEL_13;
        }
        v8 = sub_BB5290(a2);
        if ( !sub_BCAC40(v8, 8) )
        {
          v9 = *(_QWORD *)(a1 + 32);
          v10 = *(_BYTE *)(a2 + 1);
          v17 = 257;
          v14 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          v11 = sub_BCB2B0(*(_QWORD **)(v9 + 72));
          v12 = sub_921130((unsigned int **)v9, v11, v14, &v15, 1, (__int64)v16, v10 >> 1);
          sub_F162A0(a1, a2, v12);
          sub_F207A0(a1, (__int64 *)a2);
        }
        v5 = v15;
      }
    }
  }
LABEL_13:
  sub_F11320((__int64)&v18);
  return v5;
}
