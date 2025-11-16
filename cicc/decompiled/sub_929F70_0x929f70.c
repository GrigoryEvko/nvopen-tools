// Function: sub_929F70
// Address: 0x929f70
//
__int64 __fastcall sub_929F70(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 v7; // rcx
  int v8; // eax
  unsigned int **v9; // r15
  bool v10; // zf
  unsigned int *v11; // rdi
  unsigned int v12; // ebx
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  _BYTE *v14; // r12
  int v15; // eax
  unsigned int **v16; // rdi
  char v17; // r9
  __int64 v19; // rax
  unsigned int *v20; // rdx
  unsigned int *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // [rsp+0h] [rbp-A0h]
  _QWORD v26[4]; // [rsp+10h] [rbp-90h] BYREF
  char v27; // [rsp+30h] [rbp-70h]
  char v28; // [rsp+31h] [rbp-6Fh]
  _QWORD v29[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v30; // [rsp+60h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
    LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
  if ( (unsigned __int8)v8 > 3u && (_BYTE)v8 != 5 && (v8 & 0xFD) != 4 )
  {
    if ( (unsigned __int8)sub_91B6F0(a4) )
    {
      v16 = *(unsigned int ***)(a1 + 8);
      v29[0] = "sub";
      v17 = 1;
      v30 = 259;
    }
    else
    {
      v16 = *(unsigned int ***)(a1 + 8);
      v17 = 0;
      v30 = 259;
      v29[0] = "sub";
    }
    return sub_929DE0(v16, (_BYTE *)a2, a3, (__int64)v29, 0, v17);
  }
  v9 = *(unsigned int ***)(a1 + 8);
  v28 = 1;
  v10 = *((_BYTE *)v9 + 108) == 0;
  v26[0] = "sub";
  v27 = 3;
  if ( !v10 )
  {
    v14 = (_BYTE *)sub_B35400((_DWORD)v9, 115, a2, (_DWORD)a3, v25, (__int64)v26, 0, 0);
    goto LABEL_11;
  }
  v11 = v9[10];
  v12 = *((_DWORD *)v9 + 26);
  v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v11 + 40LL);
  if ( v13 == sub_928A40 )
  {
    if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
      goto LABEL_23;
    if ( (unsigned __int8)sub_AC47B0(16) )
      v14 = (_BYTE *)sub_AD5570(16, a2, a3, 0, 0);
    else
      v14 = (_BYTE *)sub_AABE40(16, a2, a3);
  }
  else
  {
    v14 = (_BYTE *)((__int64 (__fastcall *)(unsigned int *, __int64, __int64, _BYTE *, _QWORD))v13)(
                     v11,
                     16,
                     a2,
                     a3,
                     v12);
  }
  if ( !v14 )
  {
    v12 = *((_DWORD *)v9 + 26);
LABEL_23:
    v30 = 257;
    v19 = sub_B504D0(16, a2, a3, v29, 0, 0);
    v20 = v9[12];
    v14 = (_BYTE *)v19;
    if ( v20 )
      sub_B99FD0(v19, 3, v20);
    sub_B45150(v14, v12);
    (*(void (__fastcall **)(unsigned int *, _BYTE *, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v9[11] + 16LL))(
      v9[11],
      v14,
      v26,
      v9[7],
      v9[8]);
    v21 = *v9;
    v22 = (__int64)&(*v9)[4 * *((unsigned int *)v9 + 2)];
    if ( *v9 != (unsigned int *)v22 )
    {
      do
      {
        v23 = *((_QWORD *)v21 + 1);
        v24 = *v21;
        v21 += 4;
        sub_B99FD0(v14, v24, v23);
      }
      while ( (unsigned int *)v22 != v21 );
    }
  }
LABEL_11:
  if ( unk_4D04700 && *v14 > 0x1Cu )
  {
    v15 = sub_B45210(v14);
    sub_B45150(v14, v15 | 1u);
  }
  return (__int64)v14;
}
