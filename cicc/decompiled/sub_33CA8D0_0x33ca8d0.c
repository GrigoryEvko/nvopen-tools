// Function: sub_33CA8D0
// Address: 0x33ca8d0
//
__int64 __fastcall sub_33CA8D0(_QWORD *a1, __int64 a2, __int64 a3, char a4, unsigned __int8 a5)
{
  unsigned int v5; // r12d
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  bool v9; // zf
  unsigned __int16 *v12; // rsi
  __int64 v13; // rcx
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // r9
  int v17; // eax
  __int64 v18; // rax
  __int64 *v19; // rsi
  char v20; // al
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-70h]
  unsigned __int8 v24; // [rsp+Bh] [rbp-65h]
  unsigned __int8 v25; // [rsp+Ch] [rbp-64h]
  unsigned int v26; // [rsp+Ch] [rbp-64h]
  unsigned int v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  _QWORD *v31; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v33[8]; // [rsp+30h] [rbp-40h] BYREF

  v7 = a1;
  v8 = *((unsigned int *)a1 + 6);
  LOBYTE(v5) = (_DWORD)v8 == 11 || (_DWORD)v8 == 35;
  if ( (_BYTE)v5 )
  {
    v9 = *(_QWORD *)(a3 + 16) == 0;
    v31 = a1;
    if ( v9 )
LABEL_29:
      sub_4263D6(a1, a2, v8);
    return (*(unsigned int (__fastcall **)(__int64, _QWORD **))(a3 + 24))(a3, &v31);
  }
  else if ( (_DWORD)v8 == 168 || (_DWORD)v8 == 156 )
  {
    v12 = (unsigned __int16 *)(a1[6] + 16LL * (unsigned int)a2);
    v8 = *v12;
    v13 = *((_QWORD *)v12 + 1);
    LOWORD(v33[0]) = v8;
    v33[1] = v13;
    if ( (_WORD)v8 )
    {
      if ( (unsigned __int16)(v8 - 17) <= 0xD3u )
      {
        v13 = 0;
        v8 = (unsigned __int16)word_4456580[(unsigned __int16)v8 - 1];
      }
    }
    else
    {
      v25 = a5;
      a1 = v33;
      v27 = v8;
      v29 = v13;
      v14 = sub_30070B0((__int64)v33);
      v13 = v29;
      v8 = v27;
      a5 = v25;
      if ( v14 )
      {
        a1 = v33;
        v21 = sub_3009970((__int64)v33, (__int64)v12, v27, v29, v25);
        a5 = v25;
        v13 = v22;
        v8 = v21;
      }
    }
    v15 = *((unsigned int *)v7 + 16);
    if ( (_DWORD)v15 )
    {
      v16 = 0;
      v30 = 40 * v15;
      do
      {
        a2 = *(_QWORD *)(v16 + v7[5]);
        v17 = *(_DWORD *)(a2 + 24);
        if ( a4 && v17 == 51 )
        {
          v9 = *(_QWORD *)(a3 + 16) == 0;
          v24 = a5;
          v23 = v16;
          v26 = v8;
          v28 = v13;
          v32 = 0;
          if ( v9 )
            goto LABEL_29;
          v19 = &v32;
        }
        else
        {
          if ( v17 != 11 && v17 != 35 )
            return v5;
          if ( !a5 )
          {
            v18 = *(_QWORD *)(a2 + 48);
            if ( (_WORD)v8 != *(_WORD *)v18 || *(_QWORD *)(v18 + 8) != v13 && !(_WORD)v8 )
              return v5;
          }
          v9 = *(_QWORD *)(a3 + 16) == 0;
          v24 = a5;
          v23 = v16;
          v26 = v8;
          v28 = v13;
          v33[0] = *(_QWORD *)(v16 + v7[5]);
          if ( v9 )
            goto LABEL_29;
          v19 = v33;
        }
        a1 = (_QWORD *)a3;
        v20 = (*(__int64 (__fastcall **)(__int64, __int64 *))(a3 + 24))(a3, v19);
        v13 = v28;
        v8 = v26;
        a5 = v24;
        if ( !v20 )
          return v5;
        v16 = v23 + 40;
      }
      while ( v30 != v23 + 40 );
    }
    return 1;
  }
  return v5;
}
