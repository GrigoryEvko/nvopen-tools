// Function: sub_6AEB80
// Address: 0x6aeb80
//
__int64 __fastcall sub_6AEB80(__int64 *a1, __int64 a2)
{
  __int64 v3; // r14
  char v4; // al
  int v5; // ebx
  __int64 v6; // rax
  bool v8; // zf
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // r14d
  __m128i *v27; // [rsp+0h] [rbp-90h]
  __int64 v28; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+18h] [rbp-78h]
  unsigned __int8 v30; // [rsp+23h] [rbp-6Dh] BYREF
  unsigned int v31; // [rsp+24h] [rbp-6Ch] BYREF
  int v32; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v33; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v34; // [rsp+30h] [rbp-60h] BYREF
  __int64 v35; // [rsp+38h] [rbp-58h] BYREF
  __int64 v36; // [rsp+40h] [rbp-50h] BYREF
  int v37; // [rsp+48h] [rbp-48h] BYREF
  __int16 v38; // [rsp+4Ch] [rbp-44h]
  int v39; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v40[7]; // [rsp+58h] [rbp-38h] BYREF

  v32 = 0;
  v30 = 0;
  if ( !(unsigned int)sub_6AD110(5, a1, &v37, &v34, &v39, v40, v27) )
    goto LABEL_9;
  LODWORD(v3) = qword_4F077B4;
  if ( (_DWORD)qword_4F077B4 )
  {
    LODWORD(v3) = 0;
  }
  else if ( !dword_4F077BC || qword_4F077A8 <= 0x9E97u )
  {
    v8 = (unsigned int)sub_6E9250(&v37) == 0;
    v4 = *(_BYTE *)(a2 + 16);
    v3 = !v8;
    if ( v4 != 1 )
      goto LABEL_5;
    goto LABEL_14;
  }
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 1 )
  {
LABEL_5:
    v29 = 0;
    if ( v4 == 2 )
    {
      v29 = *(_QWORD *)(a2 + 288);
      if ( !v29 && *(_BYTE *)(a2 + 317) == 12 && *(_BYTE *)(a2 + 320) == 1 )
        v29 = sub_72E9A0(a2 + 144);
    }
    v5 = sub_8D32E0(v34);
    if ( !v5 )
      goto LABEL_7;
LABEL_15:
    sub_6F6C80(a2);
    if ( !(unsigned int)sub_69A8F0((__int64 *)a2, v34, 0, &v39, &v30) )
      goto LABEL_8;
    v28 = *(_QWORD *)a2;
    v36 = *(_QWORD *)a2;
    v35 = v34;
    if ( (_DWORD)v3 == 1 )
      goto LABEL_9;
    sub_68F0D0(v34, (_QWORD *)a2, 0, dword_4F077BC == 0, 5u, &v39, &v35, &v36, &v32);
    goto LABEL_18;
  }
LABEL_14:
  v29 = *(_QWORD *)(a2 + 144);
  v5 = sub_8D32E0(v34);
  if ( v5 )
    goto LABEL_15;
LABEL_7:
  sub_6F69D0(a2, 0);
  if ( !(unsigned int)sub_69A8F0((__int64 *)a2, v34, 0, &v39, &v30) )
  {
LABEL_8:
    v36 = *(_QWORD *)a2;
    v35 = v34;
LABEL_9:
    sub_6E6840(a2);
    goto LABEL_10;
  }
  v28 = *(_QWORD *)a2;
  v36 = *(_QWORD *)a2;
  v35 = v34;
LABEL_18:
  if ( (_DWORD)v3 )
    goto LABEL_9;
  if ( !v32 )
  {
    v11 = v35;
    if ( (unsigned int)sub_8DB6B0(v36, v35, &v31) )
    {
      v19 = v36;
      if ( (unsigned int)sub_8DF7D0(v36, v35, &v33) )
      {
        v24 = dword_4F077BC;
        if ( !dword_4F077BC
          || (v19 = v35, !(unsigned int)sub_8D2E30(v35))
          || (v19 = sub_8D46C0(v35), !(unsigned int)sub_8D2310(v19)) )
        {
          if ( (unsigned int)sub_6E5430(v19, v24, v20, v21, v22, v23) )
            sub_6851A0(0x2B6u, &v37, (__int64)"reinterpret_cast");
        }
      }
      else
      {
        v26 = v31;
        if ( v31 || (v26 = v33, (v31 = v33) != 0) )
        {
          if ( (unsigned int)sub_6E53E0(5, v26, &v37) )
            sub_684B30(v26, &v37);
        }
      }
      if ( (dword_4F04C44 != -1
         || (v25 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v25 + 6) & 6) != 0)
         || *(_BYTE *)(v25 + 4) == 12)
        && ((unsigned int)sub_8DBE70(v28) || (unsigned int)sub_8DBE70(v34)) )
      {
        sub_6F4200(a2, v34, 5, 0);
      }
      else if ( v5 )
      {
        sub_6FAB30(a2, v34, 1, 0, 1);
      }
      else
      {
        sub_6FB850(v34, a2, (unsigned int)&v39, 0, 0, 0, 1, 1);
      }
    }
    else
    {
      if ( !dword_4F077BC || (v11 = v34, v28 != v34) && !(unsigned int)sub_8D97D0(v28, v34, 0, v12, v13) )
      {
        v14 = v34;
        if ( (unsigned int)sub_8D3A70(v34) )
        {
          if ( (unsigned int)sub_6E5430(v14, v11, v15, v16, v17, v18) )
            sub_685360(0x77u, &v39, v34);
        }
        else if ( (unsigned int)sub_6E5430(v14, v11, v15, v16, v17, v18) )
        {
          sub_6851C0(0xABu, &v37);
        }
        goto LABEL_9;
      }
      if ( (unsigned int)sub_6E53E0(5, 1399, &v37) )
        sub_684B30(0x577u, &v37);
    }
  }
  v9 = sub_6E3FE0(v29, 5, a2);
  v10 = v9;
  if ( v9 )
  {
    if ( (unsigned int)sub_730740(v9) )
      *(_BYTE *)(v10 + 58) |= 2u;
    sub_6E3CB0(v10, &v37, &v39, v34);
  }
LABEL_10:
  *(_DWORD *)(a2 + 68) = v37;
  *(_WORD *)(a2 + 72) = v38;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  v6 = v40[0];
  *(_QWORD *)(a2 + 76) = v40[0];
  unk_4F061D8 = v6;
  sub_6E3280(a2, &v37);
  return sub_6E26D0(v30, a2);
}
