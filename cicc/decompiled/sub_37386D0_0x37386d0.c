// Function: sub_37386D0
// Address: 0x37386d0
//
void __fastcall sub_37386D0(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  bool v7; // zf
  int *v8; // rbx
  __int64 v9; // rax
  int *v10; // rdx
  signed __int64 v11; // rax
  int *v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  int v17; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int8 v21; // dl
  __int64 v22; // rax
  _BYTE *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // r15
  char *v26; // rax
  __int64 v27; // rax
  char *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-F8h] BYREF
  unsigned __int64 *v30; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v31; // [rsp+18h] [rbp-E8h]
  _QWORD v32[2]; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int64 *v33; // [rsp+30h] [rbp-D0h] BYREF
  __int64 *v34; // [rsp+38h] [rbp-C8h]
  _QWORD v35[3]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE *v36; // [rsp+58h] [rbp-A8h]
  _BYTE v37[72]; // [rsp+68h] [rbp-98h] BYREF
  __int64 **v38; // [rsp+B0h] [rbp-50h]

  v6 = *a2;
  v7 = *(_BYTE *)(*a2 + 72LL) == 0;
  v8 = *(int **)(*a2 + 8LL);
  v29 = *a2;
  if ( !v7 )
  {
    v9 = 6LL * *(unsigned int *)(v6 + 16);
    v10 = &v8[v9];
    v11 = 0xAAAAAAAAAAAAAAABLL * ((v9 * 4) >> 3);
    if ( v11 >> 2 )
    {
      v12 = &v8[24 * (v11 >> 2)];
      while ( *v8 || v8[5] )
      {
        if ( !v8[6] && !v8[11] )
        {
          v8 += 6;
          break;
        }
        if ( !v8[12] && !v8[17] )
        {
          v8 += 12;
          break;
        }
        if ( !v8[18] && !v8[23] )
        {
          v8 += 18;
          break;
        }
        v8 += 24;
        if ( v8 == v12 )
        {
          v11 = 0xAAAAAAAAAAAAAAABLL * (((char *)v10 - (char *)v8) >> 3);
          goto LABEL_26;
        }
      }
LABEL_6:
      if ( v10 != v8 )
        return;
LABEL_7:
      v13 = a2[1];
      v14 = sub_A777F0(0x10u, a1 + 11);
      if ( v14 )
      {
        *(_QWORD *)v14 = 0;
        *(_DWORD *)(v14 + 8) = 0;
      }
      sub_3247620((__int64)v35, a1[23], (__int64)a1, v14);
      sub_3243D60(v35, v13);
      v30 = 0;
      v31 = 0;
      if ( v13 )
      {
        v30 = *(unsigned __int64 **)(v13 + 16);
        v31 = *(_QWORD *)(v13 + 24);
      }
      v15 = *(_QWORD *)(*(_QWORD *)(a1[23] + 232) + 16LL);
      v32[1] = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 200LL))(v15);
      v33 = v32;
      v32[0] = v35;
      v34 = &v29;
      if ( !(unsigned __int8)sub_3244260(
                               v35,
                               &v30,
                               (__int64 (__fastcall *)(__int64, unsigned __int64, unsigned __int64 **))sub_3735570,
                               (__int64)&v33) )
      {
LABEL_12:
        v16 = (unsigned __int64)v36;
        if ( v36 == v37 )
          return;
        goto LABEL_13;
      }
LABEL_45:
      sub_3243D40((__int64)v35);
      sub_3249620(a1, a4, 2, v38);
      if ( v37[63] )
      {
        LODWORD(v33) = 65547;
        sub_3249A20(a1, (unsigned __int64 **)(a4 + 8), 15875, 65547, v37[62]);
      }
      goto LABEL_12;
    }
LABEL_26:
    if ( v11 != 2 )
    {
      if ( v11 != 3 )
      {
        if ( v11 != 1 )
          goto LABEL_7;
        goto LABEL_29;
      }
      if ( !*v8 && !v8[5] )
        goto LABEL_6;
      v8 += 6;
    }
    if ( !*v8 && !v8[5] )
      goto LABEL_6;
    v8 += 6;
LABEL_29:
    if ( *v8 || v8[5] )
      goto LABEL_7;
    goto LABEL_6;
  }
  v17 = *v8;
  if ( !*v8 )
  {
    sub_37385E0(a1, a3, a4, ((unsigned __int64)(unsigned int)v8[5] << 32) | *((unsigned __int8 *)v8 + 16));
    return;
  }
  switch ( v17 )
  {
    case 1:
      v25 = a2[1];
      if ( !v25 || !(unsigned int)((__int64)(*(_QWORD *)(v25 + 24) - *(_QWORD *)(v25 + 16)) >> 3) )
      {
        v26 = (char *)sub_321DF00(a3);
        sub_3249EC0(a1, a4, *((_QWORD *)v8 + 1), v26);
        return;
      }
      v27 = sub_A777F0(0x10u, a1 + 11);
      if ( v27 )
      {
        *(_QWORD *)v27 = 0;
        *(_DWORD *)(v27 + 8) = 0;
      }
      sub_3247620((__int64)v35, a1[23], (__int64)a1, v27);
      sub_3243D60(v35, v25);
      sub_3243300((__int64)v35, *((_QWORD *)v8 + 1));
      v33 = *(unsigned __int64 **)(v25 + 16);
      v34 = *(__int64 **)(v25 + 24);
      sub_3244870(v35, &v33);
      goto LABEL_45;
    case 2:
      sub_324A320(a1, a4, *((_QWORD *)v8 + 1));
      return;
    case 3:
      v28 = (char *)sub_321DF00(a3);
      sub_324A3E0(a1, a4, *((_QWORD *)v8 + 1), v28);
      return;
    case 4:
      v19 = sub_A777F0(0x10u, a1 + 11);
      if ( v19 )
      {
        *(_QWORD *)v19 = 0;
        *(_DWORD *)(v19 + 8) = 0;
      }
      sub_3247620((__int64)v35, a1[23], (__int64)a1, v19);
      v20 = *(_QWORD *)(a3 + 8);
      v21 = *(_BYTE *)(v20 - 16);
      if ( (v21 & 2) != 0 )
        v22 = *(_QWORD *)(v20 - 32);
      else
        v22 = v20 - 16 - 8LL * ((v21 >> 2) & 0xF);
      v23 = *(_BYTE **)(v22 + 24);
      v24 = a1[23];
      if ( *v23 != 12 )
        v23 = 0;
      sub_32200A0(v24, (__int64)v23, v29, (unsigned __int64)v35);
      sub_3243D40((__int64)v35);
      sub_3249620(a1, a4, 2, v38);
      v16 = (unsigned __int64)v36;
      if ( v36 != v37 )
LABEL_13:
        _libc_free(v16);
      break;
  }
}
