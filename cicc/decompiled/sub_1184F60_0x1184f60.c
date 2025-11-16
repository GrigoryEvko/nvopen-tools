// Function: sub_1184F60
// Address: 0x1184f60
//
unsigned __int8 *__fastcall sub_1184F60(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rbx
  _BYTE *v3; // r15
  unsigned __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  int v9; // edx
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  char v14; // al
  unsigned __int64 v15; // rax
  int v16; // edx
  __int64 v17; // r14
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  int v20; // edx
  __int64 v21; // r15
  __int64 v22; // rdi
  unsigned __int8 *result; // rax
  _BYTE *v24; // rdi
  unsigned __int8 *v25; // r15
  bool v26; // al
  __int64 v27; // r15
  unsigned __int64 v28; // rax
  int v29; // edx
  unsigned __int64 v30; // rax
  int v31; // edx
  unsigned __int8 **v32; // rax
  bool v33; // al
  unsigned __int8 *v34; // [rsp+0h] [rbp-90h]
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+8h] [rbp-88h]
  _DWORD v38[6]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v39; // [rsp+28h] [rbp-68h]
  int v40; // [rsp+30h] [rbp-60h]
  __int64 v41; // [rsp+34h] [rbp-5Ch]
  int v42; // [rsp+3Ch] [rbp-54h]
  __int64 v43[2]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int8 **v44; // [rsp+50h] [rbp-40h]

  v2 = *(_BYTE **)(a2 - 96);
  v43[0] = 32;
  v3 = *(_BYTE **)(a2 - 64);
  if ( *v2 != 82 )
    return 0;
  v5 = sub_B53900((__int64)v2);
  v41 = sub_B53630(v5, v43[0]);
  v42 = v6;
  if ( !(_BYTE)v6 )
    return 0;
  v35 = *((_QWORD *)v2 - 8);
  if ( !v35 )
    return 0;
  v7 = *((_QWORD *)v2 - 4);
  if ( !v7 )
    return 0;
  v43[0] = 32;
  if ( *v3 != 82 )
    return 0;
  if ( (v8 = sub_B53900((__int64)v3), *(_QWORD *)&v38[3] = sub_B53630(v8, v43[0]), v38[5] = v9, (_BYTE)v9)
    && *((_QWORD *)v3 - 8)
    && v7 == *((_QWORD *)v3 - 4)
    || (v10 = sub_B53930(v43[0]), v11 = sub_B53900((__int64)v3), v41 = sub_B53630(v11, v10), v42 = v12, (_BYTE)v12)
    && *((_QWORD *)v3 - 4)
    && v7 == *((_QWORD *)v3 - 8) )
  {
    v13 = v7;
    v14 = *v3;
    v7 = v35;
    v35 = v13;
  }
  else
  {
    v14 = *v3;
  }
  v43[0] = 32;
  if ( v14 != 82 )
    return 0;
  v15 = sub_B53900((__int64)v3);
  v39 = sub_B53630(v15, v43[0]);
  v40 = v16;
  if ( !(_BYTE)v16 || (v17 = *((_QWORD *)v3 - 8)) == 0 || *((_QWORD *)v3 - 4) != v35 )
  {
    v18 = sub_B53930(v43[0]);
    v19 = sub_B53900((__int64)v3);
    v41 = sub_B53630(v19, v18);
    v42 = v20;
    if ( !(_BYTE)v20 )
      return 0;
    v17 = *((_QWORD *)v3 - 4);
    if ( !v17 || *((_QWORD *)v3 - 8) != v35 )
      return 0;
  }
  v43[1] = (__int64)v3;
  v21 = *(_QWORD *)(a2 - 32);
  v43[0] = 0;
  v44 = (unsigned __int8 **)v38;
  if ( *(_BYTE *)v21 <= 0x1Cu )
    return 0;
  v22 = *(_QWORD *)(v21 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
    v22 = **(_QWORD **)(v22 + 16);
  if ( !sub_BCAC40(v22, 1) )
    return 0;
  if ( *(_BYTE *)v21 == 57 )
  {
    if ( (*(_BYTE *)(v21 + 7) & 0x40) != 0 )
      v32 = *(unsigned __int8 ***)(v21 - 8);
    else
      v32 = (unsigned __int8 **)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
    v25 = *v32;
    v34 = v32[4];
    v33 = sub_9987C0((__int64)v43, 30, *v32);
    if ( v34 && v33 )
    {
      *v44 = v34;
      goto LABEL_35;
    }
    if ( sub_9987C0((__int64)v43, 30, v34) && v25 )
      goto LABEL_34;
    return 0;
  }
  if ( *(_BYTE *)v21 != 86 )
    return 0;
  v36 = *(_QWORD *)(v21 - 96);
  if ( *(_QWORD *)(v36 + 8) != *(_QWORD *)(v21 + 8) )
    return 0;
  v24 = *(_BYTE **)(v21 - 32);
  if ( *v24 > 0x15u )
    return 0;
  v25 = *(unsigned __int8 **)(v21 - 64);
  if ( !sub_AC30F0((__int64)v24) )
    return 0;
  v26 = sub_9987C0((__int64)v43, 30, (unsigned __int8 *)v36);
  if ( v25 && v26 )
  {
LABEL_34:
    *v44 = v25;
    goto LABEL_35;
  }
  if ( !sub_9987C0((__int64)v43, 30, v25) )
    return 0;
  *v44 = (unsigned __int8 *)v36;
LABEL_35:
  v27 = *(_QWORD *)v38;
  v43[0] = 32;
  result = 0;
  if ( **(_BYTE **)v38 == 82 )
  {
    v28 = sub_B53900(*(__int64 *)v38);
    v41 = sub_B53630(v28, v43[0]);
    v42 = v29;
    if ( (_BYTE)v29 && *(_QWORD *)(v27 - 64) == v7 && *(_QWORD *)(v27 - 32) == v17
      || (v37 = sub_B53930(v43[0]), v30 = sub_B53900(v27), v41 = sub_B53630(v30, v37), v42 = v31, (_BYTE)v31)
      && *(_QWORD *)(v27 - 32) == v7
      && v17 == *(_QWORD *)(v27 - 64) )
    {
      *(_BYTE *)(*(_QWORD *)v38 + 1LL) &= ~2u;
      return sub_F162A0(a1, a2, *(__int64 *)v38);
    }
    return 0;
  }
  return result;
}
