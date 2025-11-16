// Function: sub_1732DB0
// Address: 0x1732db0
//
__int64 __fastcall sub_1732DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  unsigned __int64 v8; // rsi
  bool v9; // zf
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int8 *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  bool v22; // bl
  int v23; // eax
  __int64 v24; // rcx
  _BYTE *v25; // r15
  __int64 v26; // rax
  unsigned __int8 v27; // al
  _QWORD *v28; // rdx
  __int64 *v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // rax
  unsigned int v32; // ebx
  __int64 v33; // rax
  int v34; // [rsp+18h] [rbp-78h]
  bool v35; // [rsp+1Ch] [rbp-74h]
  unsigned int v36; // [rsp+1Ch] [rbp-74h]
  int v37; // [rsp+1Ch] [rbp-74h]
  __int64 v38; // [rsp+20h] [rbp-70h] BYREF
  __int64 v39; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v40[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v41[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v42; // [rsp+50h] [rbp-40h]

  v8 = *(_QWORD *)(a1 - 48);
  v9 = *(_BYTE *)(a1 + 16) == 50;
  v40[0] = &v38;
  v10 = *(_QWORD *)(v8 + 8);
  v11 = v9 + 26;
  if ( !v10 )
    return 0;
  if ( *(_QWORD *)(v10 + 8) )
    return 0;
  if ( !sub_171DA10(v40, v8, a3, a4) )
    return 0;
  v15 = *(_QWORD *)(a1 - 24);
  v41[0] = &v39;
  v16 = *(_QWORD *)(v15 + 8);
  if ( !v16 || *(_QWORD *)(v16 + 8) || !sub_171DA10(v41, v15, v13, v14) )
    return 0;
  v19 = (unsigned __int8 *)v38;
  v20 = *(_QWORD *)(v38 + 8);
  v35 = v20 && *(_QWORD *)(v20 + 8) == 0;
  v22 = sub_15FB730(v38, v15, v17, v18);
  if ( v22 )
    return 0;
  v23 = v19[16];
  if ( (_BYTE)v23 == 13 )
    return 0;
  v24 = *(_QWORD *)v19;
  if ( *(_BYTE *)(*(_QWORD *)v19 + 8LL) == 16 && (unsigned __int8)v23 <= 0x10u )
  {
    v34 = *(_QWORD *)(v24 + 32);
    if ( !v34 )
      return 0;
    v30 = 0;
    while ( 1 )
    {
      v15 = v30;
      v36 = v30;
      v31 = sub_15A0A60((__int64)v19, v30);
      if ( !v31 || (*(_BYTE *)(v31 + 16) & 0xFB) != 9 )
        break;
      v30 = v36 + 1;
      if ( v34 == v36 + 1 )
        return 0;
    }
  }
  else if ( (unsigned __int8)v23 > 0x17u )
  {
    v24 = (unsigned int)(v23 - 75);
    if ( (unsigned __int8)(v23 - 75) <= 1u
      || (v24 = (unsigned int)(unsigned __int8)v23 - 35, (unsigned int)v24 <= 0x11)
      && (((_BYTE)v23 - 35) & 0xFD) == 0
      && (*(_BYTE *)(*((_QWORD *)v19 - 6) + 16LL) <= 0x10u || *(_BYTE *)(*((_QWORD *)v19 - 3) + 16LL) <= 0x10u) )
    {
      if ( v35 )
        return 0;
    }
  }
  v25 = (_BYTE *)v39;
  v26 = *(_QWORD *)(v39 + 8);
  if ( v26 )
    v22 = *(_QWORD *)(v26 + 8) == 0;
  if ( sub_15FB730(v39, v15, v21, v24) )
    return 0;
  v27 = v25[16];
  if ( v27 == 13 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) == 16 && v27 <= 0x10u )
  {
    v37 = *(_QWORD *)(*(_QWORD *)v25 + 32LL);
    if ( v37 )
    {
      v32 = 0;
      while ( 1 )
      {
        v33 = sub_15A0A60((__int64)v25, v32);
        if ( !v33 || (*(_BYTE *)(v33 + 16) & 0xFB) != 9 )
          break;
        if ( v37 == ++v32 )
          return 0;
      }
      goto LABEL_26;
    }
    return 0;
  }
  if ( v27 > 0x17u
    && ((unsigned __int8)(v27 - 75) <= 1u
     || (unsigned int)v27 - 35 <= 0x11
     && ((v27 - 35) & 0xFD) == 0
     && (*(_BYTE *)(*((_QWORD *)v25 - 6) + 16LL) <= 0x10u || *(_BYTE *)(*((_QWORD *)v25 - 3) + 16LL) <= 0x10u))
    && v22 )
  {
    return 0;
  }
LABEL_26:
  v40[0] = sub_1649960(a1);
  v42 = 773;
  v40[1] = v28;
  v41[0] = v40;
  v41[1] = ".demorgan";
  v29 = (__int64 *)sub_17066B0(a2, v11, v38, v39, (__int64 *)v41, 0, a5, a6, a7);
  v42 = 257;
  return sub_15FB630(v29, (__int64)v41, 0);
}
