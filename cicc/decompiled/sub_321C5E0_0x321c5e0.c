// Function: sub_321C5E0
// Address: 0x321c5e0
//
__int64 __fastcall sub_321C5E0(
        _QWORD **a1,
        unsigned int a2,
        unsigned int a3,
        _BYTE *a4,
        unsigned int a5,
        unsigned int a6,
        unsigned __int16 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  bool v14; // zf
  unsigned __int16 v15; // si
  bool v16; // r13
  char v17; // dl
  _BYTE *v18; // r8
  unsigned __int8 v19; // al
  unsigned __int8 v20; // di
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r15
  const char *v27; // r14
  __int64 v28; // rdi
  unsigned int v29; // r9d
  _QWORD *v30; // rcx
  _BYTE *v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // r9
  void (*v35)(); // r13
  unsigned int v36; // [rsp+Ch] [rbp-44h]
  unsigned int v37; // [rsp+Ch] [rbp-44h]
  _BYTE *v38; // [rsp+10h] [rbp-40h]
  unsigned int v39; // [rsp+10h] [rbp-40h]
  unsigned int v40; // [rsp+10h] [rbp-40h]
  unsigned int v41; // [rsp+18h] [rbp-38h]
  unsigned int v42; // [rsp+18h] [rbp-38h]
  unsigned int v43; // [rsp+18h] [rbp-38h]
  unsigned int v44; // [rsp+1Ch] [rbp-34h]
  unsigned int v45; // [rsp+1Ch] [rbp-34h]
  unsigned int v46; // [rsp+1Ch] [rbp-34h]

  v14 = a2 == 0;
  v15 = a7;
  v16 = !v14;
  if ( !a4 )
  {
    v27 = 0;
    v26 = 0;
    v33 = 0;
    v32 = 1;
    goto LABEL_17;
  }
  v17 = *a4;
  v18 = a4;
  if ( *a4 == 16 )
    goto LABEL_5;
  v19 = *(a4 - 16);
  if ( (v19 & 2) != 0 )
  {
    v18 = (_BYTE *)**((_QWORD **)a4 - 4);
    if ( !v18 )
      goto LABEL_22;
LABEL_5:
    v20 = *(v18 - 16);
    if ( (v20 & 2) != 0 )
      v21 = (__int64 *)*((_QWORD *)v18 - 4);
    else
      v21 = (__int64 *)&v18[-8 * ((v20 >> 2) & 0xF) - 16];
    v22 = *v21;
    if ( *v21 )
    {
      v36 = a6;
      v41 = a5;
      v38 = a4;
      v44 = a3;
      v23 = sub_B91420(v22);
      a4 = v38;
      a3 = v44;
      v22 = v23;
      a5 = v41;
      v25 = v24;
      a6 = v36;
      v17 = *v38;
      v15 = a7;
    }
    else
    {
      v25 = 0;
    }
    v26 = v25;
    v27 = (const char *)v22;
    v28 = *(_QWORD *)(a8 + 8LL * a6);
    if ( v15 <= 3u || !v16 )
      goto LABEL_25;
    goto LABEL_11;
  }
  v18 = *(_BYTE **)&a4[-8 * ((v19 >> 2) & 0xF) - 16];
  if ( v18 )
    goto LABEL_5;
LABEL_22:
  v27 = byte_3F871B3;
  v26 = 0;
  v28 = *(_QWORD *)(a8 + 8LL * a6);
  if ( a7 <= 3u || v14 )
  {
    v29 = 0;
    goto LABEL_13;
  }
LABEL_11:
  if ( v17 == 20 )
  {
    v29 = *((_DWORD *)a4 + 1);
    v19 = *(a4 - 16);
LABEL_13:
    if ( (v19 & 2) != 0 )
      v30 = (_QWORD *)*((_QWORD *)a4 - 4);
    else
      v30 = &a4[-8 * ((v19 >> 2) & 0xF) - 16];
    v31 = (_BYTE *)*v30;
    goto LABEL_16;
  }
LABEL_25:
  if ( v17 != 16 )
  {
    v19 = *(a4 - 16);
    v29 = 0;
    goto LABEL_13;
  }
  v31 = a4;
  v29 = 0;
LABEL_16:
  v42 = a5;
  v39 = a3;
  v45 = v29;
  v32 = sub_373B2C0(v28, v31);
  v33 = v45;
  a3 = v39;
  a5 = v42;
LABEL_17:
  if ( LOBYTE(qword_5036408[8]) )
  {
    if ( v16 )
    {
      v35 = (void (*)())(*a1)[39];
      if ( v35 != nullsub_1832 )
      {
        v37 = a5;
        v43 = a3;
        v40 = v33;
        v46 = v32;
        ((void (__fastcall *)(_QWORD **, const char *, __int64, _QWORD, _QWORD))v35)(a1, v27, v26, a2, 0);
        a5 = v37;
        a3 = v43;
        v33 = v40;
        v32 = v46;
      }
    }
  }
  return (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64, const char *, __int64, __int64, __int64))(*a1[28] + 688LL))(
           a1[28],
           v32,
           a2,
           a3,
           a5,
           0,
           v33,
           v27,
           v26,
           a9,
           a10);
}
