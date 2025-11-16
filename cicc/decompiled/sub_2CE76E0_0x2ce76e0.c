// Function: sub_2CE76E0
// Address: 0x2ce76e0
//
_BYTE *__fastcall sub_2CE76E0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 **v9; // r13
  unsigned __int8 v10; // al
  unsigned __int64 v11; // r15
  __int64 v12; // rsi
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v14; // r14
  __int64 v16; // rdi
  int v17; // r13d
  char *v18; // rbx
  char *v19; // r13
  __int64 v20; // rdx
  unsigned int v21; // esi
  char v23[32]; // [rsp+10h] [rbp-120h] BYREF
  __int16 v24; // [rsp+30h] [rbp-100h]
  char v25[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+60h] [rbp-D0h]
  char *v27; // [rsp+70h] [rbp-C0h] BYREF
  int v28; // [rsp+78h] [rbp-B8h]
  char v29; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+A8h] [rbp-88h]
  __int64 v31; // [rsp+B0h] [rbp-80h]
  __int64 v32; // [rsp+C0h] [rbp-70h]
  __int64 v33; // [rsp+C8h] [rbp-68h]
  __int64 v34; // [rsp+D0h] [rbp-60h]
  int v35; // [rsp+D8h] [rbp-58h]
  void *v36; // [rsp+F0h] [rbp-40h]

  v7 = (__int64 *)sub_BD5C60(a2);
  v8 = sub_BCE3C0(v7, a3);
  if ( v8 == *(_QWORD *)(a2 + 8) )
    return (_BYTE *)a2;
  v9 = (__int64 **)v8;
  v10 = *(_BYTE *)a2;
  v11 = a2;
  if ( *(_BYTE *)a2 == 79 )
  {
    v11 = *(_QWORD *)(a2 - 32);
    v14 = (_BYTE *)v11;
    if ( v9 == *(__int64 ***)(v11 + 8) )
      return v14;
    v10 = *(_BYTE *)v11;
  }
  if ( v10 <= 0x1Cu )
  {
    v16 = *(_QWORD *)(a4 + 80);
    if ( v16 )
      v16 -= 24;
    v12 = sub_AA4FF0(v16);
    if ( v12 )
      v12 -= 24;
  }
  else
  {
    v12 = sub_B46B10(v11, 0);
  }
  sub_23D0AB0((__int64)&v27, v12, 0, 0, 0);
  v24 = 257;
  if ( v9 == *(__int64 ***)(v11 + 8) )
  {
    v14 = (_BYTE *)v11;
    goto LABEL_11;
  }
  v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v32 + 120LL);
  if ( v13 == sub_920130 )
  {
    if ( *(_BYTE *)v11 > 0x15u )
    {
LABEL_22:
      v26 = 257;
      v14 = (_BYTE *)sub_B51D30(50, v11, (__int64)v9, (__int64)v25, 0, 0);
      if ( (unsigned __int8)sub_920620((__int64)v14) )
      {
        v17 = v35;
        if ( v34 )
          sub_B99FD0((__int64)v14, 3u, v34);
        sub_B45150((__int64)v14, v17);
      }
      (*(void (__fastcall **)(__int64, _BYTE *, char *, __int64, __int64))(*(_QWORD *)v33 + 16LL))(
        v33,
        v14,
        v23,
        v30,
        v31);
      v18 = v27;
      v19 = &v27[16 * v28];
      if ( v27 != v19 )
      {
        do
        {
          v20 = *((_QWORD *)v18 + 1);
          v21 = *(_DWORD *)v18;
          v18 += 16;
          sub_B99FD0((__int64)v14, v21, v20);
        }
        while ( v19 != v18 );
      }
      goto LABEL_11;
    }
    if ( (unsigned __int8)sub_AC4810(0x32u) )
      v14 = (_BYTE *)sub_ADAB70(50, v11, v9, 0);
    else
      v14 = (_BYTE *)sub_AA93C0(0x32u, v11, (__int64)v9);
  }
  else
  {
    v14 = (_BYTE *)v13(v32, 50u, (_BYTE *)v11, (__int64)v9);
  }
  if ( !v14 )
    goto LABEL_22;
LABEL_11:
  if ( *v14 > 0x1Cu )
    sub_2CE7030(a1, a2, (__int64)v14);
  nullsub_61();
  v36 = &unk_49DA100;
  nullsub_63();
  if ( v27 != &v29 )
    _libc_free((unsigned __int64)v27);
  return v14;
}
