// Function: sub_1234A00
// Address: 0x1234a00
//
__int64 __fastcall sub_1234A00(__int64 a1, _QWORD *a2, __int64 *a3, int a4)
{
  unsigned int v6; // r13d
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  int v10; // ecx
  unsigned __int8 v11; // dl
  const char *v12; // rax
  _QWORD *v13; // rbx
  __int64 v14; // r15
  __int16 v15; // r12
  __int64 v16; // rdx
  int v17; // ecx
  int v18; // eax
  _QWORD *v19; // rdi
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r15
  __int16 v23; // r12
  __int64 v24; // rdx
  int v25; // ecx
  int v26; // eax
  _QWORD *v27; // rdi
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+8h] [rbp-98h]
  int v32; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v33; // [rsp+20h] [rbp-80h] BYREF
  __int64 v34; // [rsp+28h] [rbp-78h] BYREF
  __int64 v35; // [rsp+30h] [rbp-70h]
  __int64 v36; // [rsp+38h] [rbp-68h]
  _QWORD v37[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v38; // [rsp+60h] [rbp-40h]

  if ( (unsigned __int8)sub_1210230(a1, &v32, a4) )
    return 1;
  v8 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v33, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after compare value") )
    return 1;
  v6 = sub_1224B80((__int64 **)a1, *(_QWORD *)(v33 + 8), &v34, a3);
  if ( (_BYTE)v6 )
    return 1;
  v9 = *(_QWORD *)(v33 + 8);
  v10 = *(unsigned __int8 *)(v9 + 8);
  v11 = *(_BYTE *)(v9 + 8);
  if ( a4 == 54 )
  {
    if ( (unsigned int)(v10 - 17) <= 1 )
      v11 = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
    if ( v11 > 3u && v11 != 5 && (v11 & 0xFD) != 4 )
    {
      HIBYTE(v38) = 1;
      v12 = "fcmp requires floating point operands";
      goto LABEL_17;
    }
    v38 = 257;
    v13 = sub_BD2C40(72, unk_3F10FD0);
    if ( v13 )
    {
      v22 = v33;
      v23 = v32;
      v24 = *(_QWORD *)(v33 + 8);
      v31 = v34;
      v25 = *(unsigned __int8 *)(v24 + 8);
      if ( (unsigned int)(v25 - 17) > 1 )
      {
        v29 = sub_BCB2A0(*(_QWORD **)v24);
      }
      else
      {
        v26 = *(_DWORD *)(v24 + 32);
        v27 = *(_QWORD **)v24;
        BYTE4(v35) = (_BYTE)v25 == 18;
        LODWORD(v35) = v26;
        v28 = (__int64 *)sub_BCB2A0(v27);
        v29 = sub_BCE1B0(v28, v35);
      }
      sub_B523C0((__int64)v13, v29, 54, v23, v22, v31, (__int64)v37, 0, 0, 0);
    }
LABEL_22:
    *a2 = v13;
    return v6;
  }
  if ( (unsigned int)(v10 - 17) > 1 )
  {
    if ( (_BYTE)v10 == 12 )
      goto LABEL_18;
  }
  else
  {
    if ( *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL) == 12 )
      goto LABEL_18;
    if ( v10 == 18 )
    {
      v11 = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
      goto LABEL_15;
    }
  }
  if ( v10 == 17 )
    v11 = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
LABEL_15:
  if ( v11 == 14 )
  {
LABEL_18:
    v38 = 257;
    v13 = sub_BD2C40(72, unk_3F10FD0);
    if ( v13 )
    {
      v14 = v33;
      v15 = v32;
      v16 = *(_QWORD *)(v33 + 8);
      v30 = v34;
      v17 = *(unsigned __int8 *)(v16 + 8);
      if ( (unsigned int)(v17 - 17) > 1 )
      {
        v21 = sub_BCB2A0(*(_QWORD **)v16);
      }
      else
      {
        v18 = *(_DWORD *)(v16 + 32);
        v19 = *(_QWORD **)v16;
        BYTE4(v36) = (_BYTE)v17 == 18;
        LODWORD(v36) = v18;
        v20 = (__int64 *)sub_BCB2A0(v19);
        v21 = sub_BCE1B0(v20, v36);
      }
      sub_B523C0((__int64)v13, v21, 53, v15, v14, v30, (__int64)v37, 0, 0, 0);
    }
    goto LABEL_22;
  }
  HIBYTE(v38) = 1;
  v12 = "icmp requires integer operands";
LABEL_17:
  v37[0] = v12;
  LOBYTE(v38) = 3;
  sub_11FD800(a1 + 176, v8, (__int64)v37, 1);
  return 1;
}
