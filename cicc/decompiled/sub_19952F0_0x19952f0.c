// Function: sub_19952F0
// Address: 0x19952f0
//
__int64 __fastcall sub_19952F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        unsigned int a5,
        __int64 a6,
        _QWORD **a7,
        __int64 a8)
{
  __int64 v9; // rdx
  __int64 v10; // r10
  __int64 v13; // r11
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned __int64 v17; // r9
  unsigned __int64 v18; // rsi
  unsigned int v19; // eax
  char v20; // al
  __int64 v21; // rax
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // rsi
  unsigned int v24; // eax
  char v25; // al
  unsigned __int8 v26; // [rsp+Ch] [rbp-44h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  v9 = (__int64)a7;
  if ( *(_DWORD *)(a2 + 32) == a5 )
  {
    v10 = *(_QWORD *)(a2 + 712);
    v13 = *(_QWORD *)(a2 + 720);
    v14 = v10;
    v15 = v13;
    if ( a5 == 2 && *(_QWORD ***)(a2 + 40) != a7 )
    {
      v26 = a4;
      v28 = *(_QWORD *)(a2 + 720);
      v31 = *(_QWORD *)(a2 + 712);
      v21 = sub_1643270(*a7);
      v15 = *(_QWORD *)(a2 + 720);
      a4 = v26;
      v9 = v21;
      v13 = v28;
      v10 = v31;
      v14 = *(_QWORD *)(a2 + 712);
    }
    if ( a3 < v14 )
    {
      v17 = v15 - a3;
      if ( !v17 )
        goto LABEL_13;
      v18 = a4;
      v19 = 1;
      if ( a5 == 3 )
      {
        v19 = a4;
        v18 = -1;
      }
      v27 = v13;
      v30 = v9;
      v20 = sub_1992C60(*(__int64 **)(a1 + 32), a5, v9, (unsigned int)a8, 0, v17, v19, v18);
      v9 = v30;
      v13 = v27;
      if ( v20 )
      {
LABEL_13:
        v10 = a3;
        goto LABEL_5;
      }
    }
    else
    {
      if ( a3 <= v15 )
      {
LABEL_5:
        *(_QWORD *)(a2 + 712) = v10;
        *(_QWORD *)(a2 + 720) = v13;
        *(_QWORD *)(a2 + 40) = v9;
        *(_DWORD *)(a2 + 48) = a8;
        return 1;
      }
      v22 = a3 - v14;
      if ( a3 == v14 )
        goto LABEL_18;
      v23 = a4;
      v24 = 1;
      if ( a5 == 3 )
      {
        v24 = a4;
        v23 = -1;
      }
      v29 = v10;
      v32 = v9;
      v25 = sub_1992C60(*(__int64 **)(a1 + 32), a5, v9, (unsigned int)a8, 0, v22, v24, v23);
      v10 = v29;
      v9 = v32;
      if ( v25 )
      {
LABEL_18:
        v13 = a3;
        goto LABEL_5;
      }
    }
  }
  return 0;
}
