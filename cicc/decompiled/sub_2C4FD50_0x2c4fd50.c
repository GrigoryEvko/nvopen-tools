// Function: sub_2C4FD50
// Address: 0x2c4fd50
//
__int64 __fastcall sub_2C4FD50(__int64 a1, int a2, unsigned int a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  int v10; // r8d
  _DWORD *v11; // rax
  __int64 **v12; // rdi
  _DWORD *v13; // r15
  __int64 v14; // rbx
  _BYTE *v15; // r11
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v21; // rbx
  _QWORD *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // rax
  _BYTE *v28; // [rsp+0h] [rbp-130h]
  __int64 v29; // [rsp+0h] [rbp-130h]
  _BYTE *v30; // [rsp+0h] [rbp-130h]
  char *v32; // [rsp+10h] [rbp-120h] BYREF
  char v33; // [rsp+30h] [rbp-100h]
  char v34; // [rsp+31h] [rbp-FFh]
  _BYTE v35[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v36; // [rsp+60h] [rbp-D0h]
  void *s; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+78h] [rbp-B8h]
  _DWORD v39[44]; // [rsp+80h] [rbp-B0h] BYREF

  v7 = a3;
  v8 = *(_QWORD *)(a1 + 8);
  s = v39;
  v9 = *(unsigned int *)(v8 + 32);
  v38 = 0x2000000000LL;
  v10 = v9;
  if ( (unsigned int)v9 > 0x20 )
  {
    sub_C8D5F0((__int64)&s, v39, v9, 4u, v9, a6);
    memset(s, 255, 4 * v9);
    v11 = s;
    LODWORD(v38) = v9;
  }
  else
  {
    if ( v9 )
    {
      v21 = 4 * v9;
      if ( v21 )
      {
        if ( (unsigned int)v21 >= 8 )
        {
          *(_QWORD *)((char *)&v39[-2] + (unsigned int)v21) = -1;
          memset(v39, 0xFFu, 8LL * ((unsigned int)(v21 - 1) >> 3));
        }
        else if ( (v21 & 4) != 0 )
        {
          v39[0] = -1;
          *(_DWORD *)((char *)&v39[-1] + (unsigned int)v21) = -1;
        }
        else if ( (_DWORD)v21 )
        {
          LOBYTE(v39[0]) = -1;
        }
      }
    }
    LODWORD(v38) = v10;
    v11 = v39;
  }
  v11[v7] = a2;
  v12 = *(__int64 ***)(a1 + 8);
  v34 = 1;
  v13 = s;
  v32 = "shift";
  v14 = (unsigned int)v38;
  v33 = 3;
  v15 = (_BYTE *)sub_ACADE0(v12);
  v16 = a4[10];
  v17 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v16 + 112LL);
  if ( v17 != sub_9B6630 )
  {
    v30 = v15;
    v27 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v17)(v16, a1, v15, v13, v14);
    v15 = v30;
    v19 = v27;
LABEL_8:
    if ( v19 )
      goto LABEL_9;
    goto LABEL_17;
  }
  if ( *(_BYTE *)a1 <= 0x15u && *v15 <= 0x15u )
  {
    v28 = v15;
    v18 = sub_AD5CE0(a1, (__int64)v15, v13, v14, 0);
    v15 = v28;
    v19 = v18;
    goto LABEL_8;
  }
LABEL_17:
  v29 = (__int64)v15;
  v36 = 257;
  v22 = sub_BD2C40(112, unk_3F1FE60);
  v19 = (__int64)v22;
  if ( v22 )
    sub_B4E9E0((__int64)v22, a1, v29, v13, v14, (__int64)v35, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)a4[11] + 16LL))(
    a4[11],
    v19,
    &v32,
    a4[7],
    a4[8]);
  v23 = *a4;
  v24 = *a4 + 16LL * *((unsigned int *)a4 + 2);
  if ( *a4 != v24 )
  {
    do
    {
      v25 = *(_QWORD *)(v23 + 8);
      v26 = *(_DWORD *)v23;
      v23 += 16;
      sub_B99FD0(v19, v26, v25);
    }
    while ( v24 != v23 );
  }
LABEL_9:
  if ( s != v39 )
    _libc_free((unsigned __int64)s);
  return v19;
}
