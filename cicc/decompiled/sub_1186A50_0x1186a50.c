// Function: sub_1186A50
// Address: 0x1186a50
//
_QWORD *__fastcall sub_1186A50(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 *a4)
{
  _QWORD *v4; // r12
  __int64 v7; // rdi
  __int64 v10; // r9
  __int16 v11; // ax
  unsigned int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rsi
  _QWORD *v15; // rax
  __int64 v16; // r13
  __int64 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned int v21; // r15d
  __int64 v22; // rax
  bool v23; // al
  int v24; // eax
  int v25; // eax
  char v26; // al
  __int64 *v27; // [rsp+8h] [rbp-A8h]
  __int64 *v28; // [rsp+8h] [rbp-A8h]
  __int64 *v29; // [rsp+8h] [rbp-A8h]
  __int64 *v30; // [rsp+10h] [rbp-A0h]
  __int64 v31; // [rsp+10h] [rbp-A0h]
  __int64 **v32; // [rsp+10h] [rbp-A0h]
  __int64 **v33; // [rsp+10h] [rbp-A0h]
  __int64 v34; // [rsp+18h] [rbp-98h]
  __int64 *v35; // [rsp+18h] [rbp-98h]
  _QWORD *v36[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v37; // [rsp+40h] [rbp-70h]
  __int64 v38; // [rsp+50h] [rbp-60h] BYREF
  __int64 v39; // [rsp+58h] [rbp-58h]
  __int16 v40; // [rsp+70h] [rbp-40h]

  v4 = *(_QWORD **)(a1 + 16);
  if ( !v4 )
    return v4;
  v4 = (_QWORD *)v4[1];
  if ( v4 )
    return 0;
  v7 = *(_QWORD *)(a1 - 32);
  v10 = v7 + 24;
  if ( *(_BYTE *)v7 != 17 )
  {
    v35 = a4;
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 > 1 )
      return v4;
    if ( *(_BYTE *)v7 > 0x15u )
      return v4;
    v20 = sub_AD7630(v7, 1, v19);
    if ( !v20 || *v20 != 17 )
      return v4;
    a4 = v35;
    v10 = (__int64)(v20 + 24);
  }
  v34 = *(_QWORD *)(a1 - 64);
  v11 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( v11 == 36 )
  {
    v21 = *(_DWORD *)(v10 + 8);
    if ( v21 > 0x40 )
    {
      v29 = a4;
      v33 = (__int64 **)v10;
      v25 = sub_C444A0(v10);
      v10 = (__int64)v33;
      a4 = v29;
      if ( v21 - v25 > 0x40 )
        return v4;
      v22 = **v33;
    }
    else
    {
      v22 = *(_QWORD *)v10;
    }
    if ( v22 != 2 )
      return v4;
    v27 = a4;
    v31 = v10;
    v38 = 0;
    v39 = v34;
    v23 = sub_99C280((__int64)&v38, 15, (unsigned __int8 *)a2);
    v10 = v31;
    a4 = v27;
    if ( v23 )
    {
      v36[0] = 0;
      v26 = sub_995B10(v36, (__int64)a3);
      a4 = v27;
      if ( v26 )
      {
        v37 = 257;
        v14 = v34;
LABEL_14:
        v15 = sub_10A0740(a4, v14, (__int64)v36);
        v16 = *(_QWORD *)(a2 + 8);
        v17 = (__int64)v15;
        v40 = 257;
        v18 = sub_BD2C40(72, unk_3F10A14);
        v4 = v18;
        if ( v18 )
          sub_B51650((__int64)v18, v17, v16, (__int64)&v38, 0, 0);
        return v4;
      }
      v10 = v31;
      v11 = *(_WORD *)(a1 + 2) & 0x3F;
    }
    else
    {
      v11 = *(_WORD *)(a1 + 2) & 0x3F;
    }
  }
  if ( v11 == 34 )
  {
    v12 = *(_DWORD *)(v10 + 8);
    if ( v12 > 0x40 )
    {
      v28 = a4;
      v32 = (__int64 **)v10;
      v24 = sub_C444A0(v10);
      a4 = v28;
      if ( v12 - v24 > 0x40 )
        return v4;
      v13 = **v32;
    }
    else
    {
      v13 = *(_QWORD *)v10;
    }
    if ( v13 == 1 )
    {
      v30 = a4;
      v38 = 0;
      v39 = v34;
      if ( sub_99C280((__int64)&v38, 15, a3) )
      {
        v36[0] = 0;
        if ( (unsigned __int8)sub_995B10(v36, a2) )
        {
          v14 = v34;
          a4 = v30;
          v37 = 257;
          goto LABEL_14;
        }
      }
    }
  }
  return v4;
}
