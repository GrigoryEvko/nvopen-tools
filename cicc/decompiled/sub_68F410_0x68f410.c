// Function: sub_68F410
// Address: 0x68f410
//
__int64 __fastcall sub_68F410(__int64 *a1, __int64 *a2, __int16 a3, int a4, _DWORD *a5, int a6)
{
  unsigned int v6; // r15d
  __int64 v9; // rdi
  bool v10; // zf
  char v11; // dl
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // r14d
  unsigned __int8 v16; // al
  __int64 v18; // rax
  char i; // dl
  int v20; // eax
  char *v21; // rdx
  __int64 *v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 j; // r14
  __int64 k; // r8
  _BOOL4 v28; // eax
  __int64 v29; // [rsp-10h] [rbp-70h]
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+0h] [rbp-60h]
  int v32; // [rsp+8h] [rbp-58h]
  __int64 *v33; // [rsp+8h] [rbp-58h]
  unsigned __int8 v36; // [rsp+23h] [rbp-3Dh] BYREF
  int v37; // [rsp+24h] [rbp-3Ch] BYREF
  _QWORD v38[7]; // [rsp+28h] [rbp-38h] BYREF

  LOWORD(v6) = a3;
  v9 = *a1;
  v10 = *((_BYTE *)a1 + 16) == 0;
  v38[0] = v9;
  if ( v10 )
    goto LABEL_6;
  v11 = *(_BYTE *)(v9 + 140);
  if ( v11 == 12 )
  {
    v12 = v9;
    do
    {
      v12 = *(_QWORD *)(v12 + 160);
      v11 = *(_BYTE *)(v12 + 140);
    }
    while ( v11 == 12 );
  }
  if ( !v11 || !*((_BYTE *)a2 + 16) )
    goto LABEL_6;
  v18 = *a2;
  for ( i = *(_BYTE *)(*a2 + 140); i == 12; i = *(_BYTE *)(v18 + 140) )
    v18 = *(_QWORD *)(v18 + 160);
  if ( i )
  {
    v6 = (unsigned __int16)v6;
    if ( (unsigned int)((__int64 (*)(void))sub_8D28B0)() || (v32 = sub_8D28B0(*a2)) != 0 )
    {
      sub_6E8B30(a1, a2, v38);
      v32 = 0;
      v13 = v38[0];
    }
    else if ( (unsigned int)sub_8D2E30(*a1) || (unsigned int)sub_8D2E30(*a2) )
    {
      sub_6EB6C0((_DWORD)a1, (_DWORD)a2, (_DWORD)a5, byte_4B6D300[(unsigned __int16)v6], 1, 1, 1, 1, (__int64)v38);
      v13 = v38[0];
    }
    else if ( (unsigned int)sub_8D3D10(*a1) || (v32 = sub_8D3D10(*a2)) != 0 )
    {
      sub_6FC4F0(a1, a2, a5, v38);
      v32 = 0;
      v13 = v38[0];
    }
    else if ( (unsigned int)sub_8D2630(*a1, a2) || (unsigned int)sub_8D2630(*a2, a2) )
    {
      if ( !(unsigned int)sub_8D2630(*a1, a2) || !(unsigned int)sub_8D2630(*a2, a2) )
        sub_6E5ED0(3374, a5, *a1, *a2);
      v38[0] = sub_72CD60();
      v13 = v38[0];
    }
    else if ( (unsigned int)sub_8D2660(*a1) || (v32 = sub_8D2660(*a2)) != 0 )
    {
      sub_6E8FF0(a1, a2, a5, v38);
      v32 = 0;
      v13 = v38[0];
    }
    else if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_6FD310((unsigned __int16)v6, a1, a2, a5, v38, &v36) )
    {
      v13 = v38[0];
    }
    else
    {
      v32 = sub_6E9580(a2);
      if ( v32 )
      {
        for ( j = *a1; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        for ( k = *a2; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        v31 = k;
        v28 = sub_68B1F0(a1, a2, &v37);
        v25 = v31;
        v32 = v28;
        if ( *(_BYTE *)(j + 140) == 2 && *(_BYTE *)(v31 + 140) == 2 )
          sub_68B0C0((_DWORD *)j, v31, (__int64)a5, 4u);
      }
      v38[0] = sub_6E8B10(a1, a2, v23, v24, v25);
      v13 = v38[0];
    }
  }
  else
  {
LABEL_6:
    v32 = 0;
    v6 = (unsigned __int16)v6;
    v38[0] = sub_72C930(v9);
    v13 = v38[0];
  }
  if ( !(unsigned int)sub_8D2B80(v13) )
  {
    v15 = sub_6EFF80();
    v36 = sub_6E9930(v6, v38[0]);
    sub_6FC7D0(v38[0], a1, a2, v36);
    if ( v32 )
    {
      v22 = a1;
      if ( v37 )
        v22 = a2;
      if ( *((_BYTE *)v22 + 16) == 2 )
      {
        v33 = v22;
        v30 = (__int64)(v22 + 18);
        if ( (unsigned int)sub_8D2930(v22[34]) )
        {
          if ( *((_BYTE *)v33 + 317) == 1 && (int)sub_6210B0(v30, 0) < 0 && (unsigned int)sub_6E53E0(5, 514, a5) )
            sub_684B30(0x202u, a5);
        }
      }
    }
    if ( !(unsigned int)sub_8D2E30(*a1)
      || !(unsigned int)sub_6E9880(a2)
      || (v20 = sub_6E98A0(a1), v21 = (char *)a1 + 68, !v20) )
    {
      if ( !(unsigned int)sub_8D2E30(*a2) || !(unsigned int)sub_6E9880(a1) || !(unsigned int)sub_6E98A0(a2) )
        goto LABEL_15;
      v21 = (char *)a2 + 68;
    }
    sub_6E5C80(4, 2330, v21);
LABEL_15:
    v16 = v36;
    goto LABEL_11;
  }
  v14 = sub_6E8E20(v38[0]);
  v15 = v14;
  if ( *(_BYTE *)(v14 + 140) == 15 )
    *(_BYTE *)(v14 + 176) |= 1u;
  v16 = sub_6E9930(v6, v38[0]);
  v36 = v16;
LABEL_11:
  sub_7016A0(v16, (_DWORD)a1, (_DWORD)a2, v15, a6, (_DWORD)a5, a4);
  return v29;
}
