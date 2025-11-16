// Function: sub_2589160
// Address: 0x2589160
//
__int64 __fastcall sub_2589160(__int64 *a1, __int64 *a2, _BYTE *a3)
{
  __int16 v5; // ax
  unsigned int v6; // r8d
  unsigned __int8 *v7; // rdi
  __int64 v8; // r14
  int v9; // eax
  __int64 v10; // r15
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int16 v14; // ax
  __int16 v15; // ax
  __int16 v16; // ax
  __int64 v18; // rdx
  __int64 v19; // rbx
  bool v20; // zf
  __int64 v21; // [rsp+8h] [rbp-58h]
  bool v22; // [rsp+17h] [rbp-49h] BYREF
  __int64 v23; // [rsp+18h] [rbp-48h] BYREF
  __int64 v24[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = sub_D139D0(a2, 0, (unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, __int64))sub_258EEC0, *a1);
  v6 = 1;
  if ( v5 )
  {
    LOBYTE(v6) = (_BYTE)v5 == 0 && HIBYTE(v5) != 0;
    if ( (_BYTE)v6 )
    {
      *a3 = 1;
      return v6;
    }
    v7 = (unsigned __int8 *)a2[3];
    v8 = a1[2];
    v9 = *v7;
    if ( (_BYTE)v9 == 76 || (_BYTE)v9 == 62 )
      goto LABEL_10;
    v10 = a1[3];
    if ( (_BYTE)v9 == 30 )
    {
      v19 = sub_B43CB0((__int64)v7);
      v20 = v19 == sub_25096F0((_QWORD *)(v10 + 72));
      v14 = *(_WORD *)(v8 + 10);
      if ( v20 )
      {
        v15 = v14 & 0xFFFB;
        goto LABEL_12;
      }
LABEL_11:
      v15 = v14 & 0xFFF8;
LABEL_12:
      v16 = *(_WORD *)(v8 + 8) | v15;
      *(_WORD *)(v8 + 10) = v16;
      LOBYTE(v6) = (v16 & 3) == 3;
      return v6;
    }
    v11 = (unsigned int)(v9 - 34);
    if ( (unsigned __int8)v11 > 0x33u
      || (v12 = 0x8000000000041LL, !_bittest64(&v12, v11))
      || a2 < (__int64 *)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)]
      || (v21 = a2[3], v13 = a1[1], a2 >= (__int64 *)sub_24E54B0(v7)) )
    {
LABEL_10:
      v14 = *(_WORD *)(v8 + 10);
      goto LABEL_11;
    }
    v24[0] = sub_254C9B0(v21, ((__int64)a2 - (v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF))) >> 5);
    v24[1] = v18;
    v23 = 0;
    if ( !(unsigned __int8)sub_25890A0(v13, v10, v24, 0, &v22, 0, &v23) )
    {
      if ( !v23 || (*(_WORD *)(v23 + 98) & 3) != 3 )
        goto LABEL_10;
      *a3 = 1;
    }
    LOBYTE(v6) = (*(_WORD *)(v8 + 10) & 3) == 3;
  }
  return v6;
}
