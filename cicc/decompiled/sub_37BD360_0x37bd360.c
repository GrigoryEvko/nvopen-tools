// Function: sub_37BD360
// Address: 0x37bd360
//
__int64 __fastcall sub_37BD360(__int64 a1, int *a2, _QWORD *a3)
{
  int v4; // r15d
  __int64 v6; // r14
  int v8; // r15d
  int v9; // eax
  int v10; // r9d
  _BYTE *v11; // rcx
  char *v12; // r8
  unsigned int i; // edx
  _BYTE *v14; // r12
  char v15; // si
  char v16; // al
  char v17; // al
  char v18; // al
  unsigned int v19; // edx
  char *v20; // [rsp+0h] [rbp-B0h]
  int v21; // [rsp+0h] [rbp-B0h]
  char *v22; // [rsp+0h] [rbp-B0h]
  int v23; // [rsp+Ch] [rbp-A4h]
  unsigned int v24; // [rsp+Ch] [rbp-A4h]
  int v25; // [rsp+Ch] [rbp-A4h]
  unsigned int v26; // [rsp+10h] [rbp-A0h]
  _BYTE *v27; // [rsp+10h] [rbp-A0h]
  unsigned int v28; // [rsp+10h] [rbp-A0h]
  _BYTE *v29; // [rsp+18h] [rbp-98h]
  char *v30; // [rsp+18h] [rbp-98h]
  _BYTE *v31; // [rsp+18h] [rbp-98h]
  _QWORD v32[6]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v33[12]; // [rsp+50h] [rbp-60h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v32[0] = 21;
  v32[2] = 0;
  v8 = v4 - 1;
  v33[0] = 22;
  v33[2] = 0;
  v9 = sub_2EAE040(a2);
  v10 = 1;
  v11 = 0;
  v12 = (char *)v32;
  for ( i = v8 & v9; ; i = v8 & v19 )
  {
    v14 = (_BYTE *)(v6 + 48LL * i);
    if ( (unsigned __int8)(*(_BYTE *)a2 - 21) > 1u )
    {
      v20 = v12;
      v23 = v10;
      v26 = i;
      v29 = v11;
      v16 = sub_2EAB6C0((__int64)a2, (char *)(v6 + 48LL * i));
      v11 = v29;
      i = v26;
      v10 = v23;
      v12 = v20;
      if ( v16 )
      {
LABEL_6:
        *a3 = v14;
        return 1;
      }
      v15 = *v14;
    }
    else
    {
      v15 = *v14;
      if ( *(_BYTE *)a2 == *v14 )
        goto LABEL_6;
    }
    if ( (unsigned __int8)(v15 - 21) <= 1u )
    {
      if ( LOBYTE(v32[0]) == v15 )
        break;
LABEL_20:
      if ( LOBYTE(v33[0]) != v15 )
        goto LABEL_19;
      goto LABEL_17;
    }
    v21 = v10;
    v24 = i;
    v27 = v11;
    v30 = v12;
    v17 = sub_2EAB6C0((__int64)v14, v12);
    v12 = v30;
    v11 = v27;
    i = v24;
    v10 = v21;
    if ( v17 )
      break;
    v15 = *v14;
    if ( (unsigned __int8)(*v14 - 21) <= 1u )
      goto LABEL_20;
    v22 = v30;
    v25 = v10;
    v28 = i;
    v31 = v11;
    v18 = sub_2EAB6C0((__int64)v14, (char *)v33);
    v11 = v31;
    i = v28;
    v10 = v25;
    v12 = v22;
    if ( !v18 )
      goto LABEL_19;
LABEL_17:
    if ( !v11 )
      v11 = v14;
LABEL_19:
    v19 = v10 + i;
    ++v10;
  }
  if ( !v11 )
    v11 = v14;
  *a3 = v11;
  return 0;
}
