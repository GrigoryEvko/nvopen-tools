// Function: sub_3987590
// Address: 0x3987590
//
__int64 __fastcall sub_3987590(
        _QWORD **a1,
        unsigned int a2,
        unsigned int a3,
        _BYTE *a4,
        unsigned int a5,
        unsigned int a6,
        unsigned __int16 a7,
        __int64 a8)
{
  unsigned __int16 v11; // r8
  char v12; // dl
  _BYTE *v13; // rsi
  __int64 v14; // rax
  _BYTE *v15; // r11
  _BYTE *v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r13
  const char *v22; // r12
  __int64 v23; // rdi
  unsigned int v24; // r11d
  unsigned int v26; // [rsp+4h] [rbp-4Ch]
  _BYTE *v27; // [rsp+8h] [rbp-48h]
  _BYTE *v28; // [rsp+10h] [rbp-40h]
  unsigned int v29; // [rsp+10h] [rbp-40h]
  bool v31; // [rsp+1Ch] [rbp-34h]
  unsigned int v32; // [rsp+1Ch] [rbp-34h]

  v11 = a7;
  v31 = a2 != 0;
  if ( !a4 )
  {
    v29 = 0;
    v22 = 0;
    v21 = 0;
    v24 = 1;
    goto LABEL_13;
  }
  v12 = *a4;
  v13 = a4;
  if ( *a4 == 15 )
  {
    v16 = a4;
    v15 = a4;
LABEL_5:
    v17 = *(_QWORD *)&v16[-8 * *((unsigned int *)v15 + 2)];
    if ( v17 )
    {
      v26 = a6;
      v27 = a4;
      v28 = a4;
      v18 = sub_161E970(v17);
      a4 = v27;
      v13 = v28;
      v17 = v18;
      a6 = v26;
      v20 = v19;
      v11 = a7;
      v12 = *v27;
    }
    else
    {
      v20 = 0;
    }
    v21 = v20;
    v22 = (const char *)v17;
    v23 = *(_QWORD *)(a8 + 8LL * a6);
    if ( v11 <= 3u || !v31 )
      goto LABEL_17;
    goto LABEL_9;
  }
  v14 = *((unsigned int *)a4 + 2);
  v15 = *(_BYTE **)&a4[-8 * v14];
  if ( v15 )
  {
    v16 = *(_BYTE **)&a4[-8 * *((unsigned int *)a4 + 2)];
    goto LABEL_5;
  }
  v22 = byte_3F871B3;
  v21 = 0;
  v23 = *(_QWORD *)(a8 + 8LL * a6);
  if ( a7 <= 3u || !v31 )
  {
    v29 = 0;
    goto LABEL_11;
  }
LABEL_9:
  if ( v12 == 19 )
  {
    v29 = *((_DWORD *)a4 + 6);
    v14 = *((unsigned int *)a4 + 2);
    goto LABEL_11;
  }
LABEL_17:
  v29 = 0;
  if ( v12 == 15 )
    goto LABEL_12;
  v14 = *((unsigned int *)a4 + 2);
LABEL_11:
  v13 = *(_BYTE **)&a4[-8 * v14];
LABEL_12:
  v24 = sub_39CC330(v23, v13);
LABEL_13:
  if ( LOBYTE(qword_5056180[20]) && v31 )
  {
    v32 = v24;
    ((void (__fastcall *)(_QWORD **, const char *, __int64, _QWORD, _QWORD))(*a1)[35])(a1, v22, v21, a2, 0);
    v24 = v32;
  }
  return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64))(*a1[32] + 584LL))(
           a1[32],
           v24,
           a2,
           a3,
           a5,
           0,
           v29,
           v22,
           v21);
}
