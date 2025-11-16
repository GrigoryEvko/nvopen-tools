// Function: sub_2A19390
// Address: 0x2a19390
//
void __fastcall sub_2A19390(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rax
  _QWORD v18[3]; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE *v19; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 (__fastcall *v21)(unsigned __int64 *, const __m128i **, int); // [rsp+30h] [rbp-90h]
  __int64 (__fastcall *v22)(); // [rsp+38h] [rbp-88h]
  __int64 v23; // [rsp+40h] [rbp-80h] BYREF
  char *v24; // [rsp+48h] [rbp-78h]
  __int64 v25; // [rsp+50h] [rbp-70h]
  int v26; // [rsp+58h] [rbp-68h]
  char v27; // [rsp+5Ch] [rbp-64h]
  char v28; // [rsp+60h] [rbp-60h] BYREF

  v18[0] = a4;
  v18[1] = a5;
  v23 = 0;
  v24 = &v28;
  v25 = 8;
  v26 = 0;
  v27 = 1;
  v21 = 0;
  v7 = (_QWORD *)sub_22077B0(0x20u);
  if ( v7 )
  {
    v7[1] = a3;
    v7[2] = v20;
    *v7 = &v23;
    v7[3] = v18;
  }
  v20[0] = v7;
  v22 = sub_2A198B0;
  v21 = sub_2A18FC0;
  v8 = sub_AA5930((__int64)a1);
  v10 = v9;
  v11 = v8;
  while ( v10 != v11 )
  {
    v12 = *(_QWORD *)(v11 - 8);
    v13 = 0x1FFFFFFFE0LL;
    v14 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
    if ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 0 )
    {
      v15 = 0;
      do
      {
        if ( a2 == *(_QWORD *)(v12 + 32LL * *(unsigned int *)(v11 + 72) + 8 * v15) )
        {
          v13 = 32 * v15;
          goto LABEL_9;
        }
        ++v15;
      }
      while ( (_DWORD)v14 != (_DWORD)v15 );
      v13 = 0x1FFFFFFFE0LL;
    }
LABEL_9:
    v16 = *(_BYTE **)(v12 + v13);
    if ( *v16 > 0x1Cu )
    {
      v19 = v16;
      if ( !v21 )
        sub_4263D6(a1, v12, v14);
      a1 = v20;
      if ( !((unsigned __int8 (__fastcall *)(_QWORD *, _BYTE **))v22)(v20, &v19) )
        break;
    }
    v17 = *(_QWORD *)(v11 + 32);
    if ( !v17 )
      BUG();
    v11 = 0;
    if ( *(_BYTE *)(v17 - 24) == 84 )
      v11 = v17 - 24;
  }
  if ( v21 )
    v21(v20, (const __m128i **)v20, 3);
  if ( !v27 )
    _libc_free((unsigned __int64)v24);
}
