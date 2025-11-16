// Function: sub_8A00C0
// Address: 0x8a00c0
//
__int64 __fastcall sub_8A00C0(__int64 a1, __int64 *a2, int a3)
{
  __int64 v4; // rbx
  char v5; // al
  __int64 *v6; // rax
  __int64 result; // rax
  char v8; // al
  char v9; // al
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // r11
  __int64 v15; // r10
  __int64 *v16; // rsi
  __int64 v17; // [rsp+0h] [rbp-80h]
  __int64 v18; // [rsp+8h] [rbp-78h]
  _QWORD *v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+10h] [rbp-70h]
  __int64 *v21; // [rsp+18h] [rbp-68h]
  int v23; // [rsp+24h] [rbp-5Ch]
  __int64 *v24; // [rsp+28h] [rbp-58h]
  __int64 *v25; // [rsp+38h] [rbp-48h] BYREF
  _QWORD *v26; // [rsp+40h] [rbp-40h] BYREF
  __int64 v27; // [rsp+48h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 88);
  v24 = *(__int64 **)(*(_QWORD *)(v4 + 104) + 176LL);
  v27 = *(_QWORD *)dword_4F07508;
  v5 = *(_BYTE *)(v4 + 160) & 0x20;
  if ( !v24 )
  {
    if ( !v5 )
      goto LABEL_7;
    goto LABEL_10;
  }
  v24 = (__int64 *)v24[2];
  if ( v5 )
  {
LABEL_10:
    v9 = *(_BYTE *)(a1 + 80);
    v25 = a2;
    if ( v9 == 20 )
    {
      v19 = **(_QWORD ***)(v4 + 328);
      sub_89A1A0(v19, a2, &v26, &v25);
    }
    else if ( v9 == 21 )
    {
      v19 = **(_QWORD ***)(v4 + 232);
      sub_89A1A0(v19, a2, &v26, &v25);
    }
    else
    {
      v19 = **(_QWORD ***)(v4 + 32);
      sub_89A1A0(v19, a2, &v26, &v25);
    }
    v10 = v26;
    if ( v26 )
    {
      v17 = v4;
      v23 = 0;
      v11 = (__int64)v19;
      v12 = 0;
      if ( a3 )
        v12 = &v27;
      v21 = v12;
      do
      {
        v13 = v25;
        if ( !v25 )
          break;
        if ( *(_BYTE *)(v10[1] + 80LL) == 3 )
        {
          v15 = v10[8];
          v14 = 0;
          if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
            v14 = *(_QWORD *)(a1 + 64);
          if ( !v23 && (*(_BYTE *)(v15 + 161) & 4) != 0 )
          {
            if ( v24 )
            {
              v18 = v10[8];
              v20 = v14;
              if ( !(unsigned int)sub_89FDA0(a1, *v24, (__int64)a2, a3, 0) )
                return 0;
              v13 = v25;
              v14 = v20;
              v23 = 1;
              v24 = 0;
              v15 = v18;
            }
            else
            {
              v23 = 1;
            }
          }
          if ( !(unsigned int)sub_89FB00(v15, v13[4], (__int64)a2, v11, v14, v21) )
            return 0;
        }
        sub_89A1C0((__int64 *)&v26, &v25);
        v10 = v26;
      }
      while ( v26 );
      v6 = v24;
      v4 = v17;
      if ( v24 )
        goto LABEL_4;
      goto LABEL_7;
    }
  }
  v6 = v24;
  if ( v24 )
  {
LABEL_4:
    result = sub_89FDA0(a1, *v6, (__int64)a2, a3, 0);
    if ( !(_DWORD)result )
      return result;
  }
LABEL_7:
  v8 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v8 - 10) > 1u && v8 != 20 )
    return 1;
  v16 = *(__int64 **)(*(_QWORD *)(v4 + 176) + 216LL);
  result = 1;
  if ( v16 )
    return (unsigned int)sub_89FDA0(a1, *v16, (__int64)a2, a3, 1) != 0;
  return result;
}
