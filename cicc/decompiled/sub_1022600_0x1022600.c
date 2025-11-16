// Function: sub_1022600
// Address: 0x1022600
//
__int64 __fastcall sub_1022600(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // rcx
  __int64 v5; // rax
  _BYTE *v6; // r13
  unsigned __int8 *v7; // r15
  int v8; // eax
  unsigned __int8 v9; // cl
  char v10; // al
  unsigned __int8 *v11; // rax
  char v12; // al
  char v13; // al
  char v14; // al
  char v15; // al
  char v16; // al
  char v17; // al
  char v18; // al
  _BYTE *v19; // [rsp+8h] [rbp-A8h]
  _BYTE *v20; // [rsp+8h] [rbp-A8h]
  _BYTE *v21; // [rsp+8h] [rbp-A8h]
  _BYTE *v22; // [rsp+8h] [rbp-A8h]
  _BYTE *v23; // [rsp+8h] [rbp-A8h]
  _BYTE *v24; // [rsp+8h] [rbp-A8h]
  _BYTE *v25; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v26; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int8 *v27; // [rsp+18h] [rbp-98h] BYREF
  _QWORD *v28[2]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v29[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v30[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v31[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD *v32[2]; // [rsp+60h] [rbp-50h] BYREF
  _QWORD *v33[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( *a3 != 86 )
    goto LABEL_3;
  v3 = *((_QWORD *)a3 - 12);
  if ( (unsigned __int8)(*(_BYTE *)v3 - 82) > 1u )
    goto LABEL_3;
  v5 = *(_QWORD *)(v3 + 16);
  if ( !v5 )
    goto LABEL_3;
  if ( *(_QWORD *)(v5 + 8) )
    goto LABEL_3;
  v6 = (_BYTE *)*((_QWORD *)a3 - 8);
  v7 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  v8 = (unsigned __int8)*v6;
  v9 = *v7;
  if ( (unsigned __int8)v8 <= 0x1Cu )
    goto LABEL_3;
  if ( (_BYTE)v8 == 84 )
  {
    if ( v9 == 84 || v9 <= 0x1Cu )
      goto LABEL_3;
    v8 = v9;
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  }
  else if ( v9 != 84 )
  {
    goto LABEL_3;
  }
  if ( (unsigned int)(v8 - 42) <= 0x11 )
  {
    v28[0] = &v26;
    v28[1] = &v27;
    if ( *v6 == 43 )
    {
      v20 = a3;
      v12 = sub_1021AA0(v28, (__int64)v6);
      a3 = v20;
      if ( v12 )
        goto LABEL_39;
    }
    v29[0] = &v26;
    v29[1] = &v27;
    if ( *v6 == 45 )
    {
      v25 = a3;
      v18 = sub_1021AA0(v29, (__int64)v6);
      a3 = v25;
      if ( v18 )
      {
LABEL_39:
        v21 = a3;
        v13 = sub_B45190((__int64)v6);
        a3 = v21;
        if ( v13 )
          goto LABEL_18;
      }
    }
    v30[0] = &v26;
    v30[1] = &v27;
    if ( *v6 == 47 )
    {
      v24 = a3;
      v16 = sub_1021AA0(v30, (__int64)v6);
      a3 = v24;
      if ( v16 )
      {
        v17 = sub_B45190((__int64)v6);
        a3 = v24;
        if ( v17 )
          goto LABEL_18;
      }
    }
    v31[0] = &v26;
    v31[1] = &v27;
    if ( *v6 == 42 )
    {
      v23 = a3;
      v15 = sub_1021AA0(v31, (__int64)v6);
      a3 = v23;
      if ( v15 )
        goto LABEL_18;
    }
    v32[0] = &v26;
    v32[1] = &v27;
    if ( *v6 == 44 )
    {
      v22 = a3;
      v14 = sub_1021AA0(v32, (__int64)v6);
      a3 = v22;
      if ( v14 )
        goto LABEL_18;
    }
    v33[0] = &v26;
    v33[1] = &v27;
    if ( *v6 == 46 )
    {
      v19 = a3;
      v10 = sub_1021AA0(v33, (__int64)v6);
      a3 = v19;
      if ( v10 )
      {
LABEL_18:
        v11 = v26;
        if ( *v26 == 84 || (v11 = v27, *v27 > 0x1Cu) )
        {
          if ( v7 == v11 )
          {
            *(_BYTE *)a1 = 1;
            *(_QWORD *)(a1 + 8) = a3;
            *(_DWORD *)(a1 + 16) = 0;
            *(_QWORD *)(a1 + 24) = 0;
            return a1;
          }
        }
      }
    }
  }
LABEL_3:
  *(_BYTE *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
