// Function: sub_E61EC0
// Address: 0xe61ec0
//
__int64 __fastcall sub_E61EC0(__int64 a1, _QWORD *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdi
  __int64 v10; // r12
  signed __int64 v11; // rax
  char *v12; // rdx
  char *v13; // rax
  _BOOL8 v14; // rsi
  char *v15; // rbx
  unsigned int v16; // r8d
  signed __int64 v17; // rdx
  signed __int64 v18; // rcx
  char *i; // rax
  char *v20; // r12
  __int64 v21; // rdx
  int v22; // r15d
  void (*v23)(); // r10
  char v24; // si
  __int64 v25; // rsi
  char *v26; // r15
  __int64 v27; // rsi
  __int64 v28; // rsi
  signed __int64 v29; // rdx
  __int64 result; // rax
  int v31; // eax
  char *v32; // rax
  __int64 v33; // [rsp+8h] [rbp-F8h]
  int v34; // [rsp+10h] [rbp-F0h]
  unsigned int v36; // [rsp+30h] [rbp-D0h]
  unsigned int v37; // [rsp+30h] [rbp-D0h]
  char *v38; // [rsp+38h] [rbp-C8h]
  char *v39; // [rsp+40h] [rbp-C0h]
  char *v40; // [rsp+48h] [rbp-B8h]
  char *v41; // [rsp+50h] [rbp-B0h] BYREF
  char *v42; // [rsp+58h] [rbp-A8h]
  __int64 v43; // [rsp+60h] [rbp-A0h]
  const char *v44; // [rsp+70h] [rbp-90h] BYREF
  char v45; // [rsp+80h] [rbp-80h]
  __int16 v46; // [rsp+90h] [rbp-70h]
  _QWORD v47[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v48; // [rsp+C0h] [rbp-40h]

  v9 = a2[1];
  v47[0] = "linetable_begin";
  v48 = 259;
  v10 = sub_E6C380(v9, v47, 0);
  v47[0] = "linetable_end";
  v48 = 259;
  v33 = sub_E6C380(v9, v47, 0);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, 242, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 832LL))(a2, v33, v10, 4);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v10, 0);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 368LL))(a2, a4, 0);
  (*(void (__fastcall **)(_QWORD *, __int64))(*a2 + 360LL))(a2, a4);
  sub_E61C70((__int64)&v41, a1, a3);
  v38 = v42;
  v39 = v41;
  v11 = 0xAAAAAAAAAAAAAAABLL * ((v42 - v41) >> 3);
  if ( v11 >> 2 <= 0 )
  {
LABEL_51:
    if ( v11 != 2 )
    {
      if ( v11 != 3 )
      {
        if ( v11 != 1 )
        {
LABEL_54:
          v14 = 0;
          v39 = v42;
          goto LABEL_9;
        }
LABEL_59:
        v32 = v39;
        if ( !*((_WORD *)v39 + 10) )
          goto LABEL_54;
LABEL_60:
        v14 = v42 != v32;
        goto LABEL_9;
      }
      v32 = v39;
      if ( *((_WORD *)v39 + 10) )
        goto LABEL_60;
      v39 += 24;
    }
    v32 = v39;
    if ( *((_WORD *)v39 + 10) )
      goto LABEL_60;
    v39 += 24;
    goto LABEL_59;
  }
  v12 = v41;
  v13 = &v41[96 * (v11 >> 2)];
  while ( 1 )
  {
    if ( *((_WORD *)v12 + 10) )
      goto LABEL_8;
    if ( *((_WORD *)v12 + 22) )
    {
      v12 += 24;
LABEL_8:
      v39 = v12;
      v14 = v42 != v12;
      goto LABEL_9;
    }
    if ( *((_WORD *)v12 + 34) )
    {
      v39 = v12 + 48;
      v14 = v42 != v12 + 48;
      goto LABEL_9;
    }
    if ( *((_WORD *)v12 + 46) )
      break;
    v12 += 96;
    if ( v13 == v12 )
    {
      v39 = v12;
      v11 = 0xAAAAAAAAAAAAAAABLL * ((v42 - v12) >> 3);
      goto LABEL_51;
    }
  }
  v39 = v12 + 72;
  v14 = v42 != v12 + 72;
LABEL_9:
  (*(void (__fastcall **)(_QWORD *, _BOOL8, __int64))(*a2 + 536LL))(a2, v14, 2);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 832LL))(a2, a5, a4, 4);
  v15 = v41;
  v40 = v42;
  if ( v41 != v42 )
  {
    while ( 1 )
    {
      v16 = *((_DWORD *)v15 + 3);
      v17 = v40 - v15;
      v18 = 0xAAAAAAAAAAAAAAABLL * ((v40 - v15) >> 3);
      if ( v18 >> 2 > 0 )
        break;
      if ( v17 == 48 )
      {
        v20 = v15;
        goto LABEL_45;
      }
      if ( v17 == 72 )
      {
        v20 = v15;
        goto LABEL_49;
      }
LABEL_47:
      v20 = v40;
LABEL_19:
      v21 = *a2;
      v22 = v18;
      v23 = *(void (**)())(*a2 + 120LL);
      v24 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + *(unsigned int *)(*(_QWORD *)(a1 + 64) + 32LL * (v16 - 1)));
      v44 = "Segment for file '";
      v46 = 2051;
      v45 = v24;
      v47[0] = &v44;
      v47[2] = "' begins";
      v48 = 770;
      if ( v23 != nullsub_98 )
      {
        v34 = v18;
        v37 = v16;
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v23)(a2, v47, 1);
        v21 = *a2;
        LODWORD(v18) = v34;
        v16 = v37;
      }
      v36 = v18;
      (*(void (__fastcall **)(_QWORD *, _QWORD))(v21 + 816))(a2, v16);
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, v36, 4);
      v25 = (unsigned int)(8 * v22 + 12);
      if ( v39 != v38 )
        v25 = (unsigned int)(v25 + 4 * v22);
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, v25, 4);
      if ( v20 != v15 )
      {
        v26 = v15;
        do
        {
          (*(void (__fastcall **)(_QWORD *, _QWORD, __int64, __int64))(*a2 + 832LL))(a2, *(_QWORD *)v26, a4, 4);
          v27 = *((unsigned int *)v26 + 4);
          if ( (v26[22] & 2) != 0 )
            v27 = *((_DWORD *)v26 + 4) | 0x80000000;
          v26 += 24;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, v27, 4);
        }
        while ( v20 != v26 );
        if ( v39 != v38 )
        {
          do
          {
            v28 = *((unsigned __int16 *)v15 + 10);
            v15 += 24;
            (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, v28, 2);
            (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, 0, 2);
          }
          while ( v20 != v15 );
        }
      }
      if ( v20 == v40 )
        goto LABEL_40;
      v15 = v20;
    }
    for ( i = v15; ; i += 96 )
    {
      if ( v16 != *((_DWORD *)i + 9) )
      {
        v20 = i + 24;
        goto LABEL_18;
      }
      if ( v16 != *((_DWORD *)i + 15) )
      {
        v20 = i + 48;
        goto LABEL_18;
      }
      if ( v16 != *((_DWORD *)i + 21) )
      {
        v20 = i + 72;
        goto LABEL_18;
      }
      v20 = i + 96;
      if ( &v15[96 * (v18 >> 2)] == i + 96 )
        break;
      if ( v16 != *((_DWORD *)i + 27) )
        goto LABEL_18;
    }
    v29 = v40 - v20;
    if ( v40 - v20 == 48 )
    {
      v31 = *((_DWORD *)i + 27);
    }
    else
    {
      if ( v29 != 72 )
      {
        if ( v29 != 24 )
        {
          v20 = v40;
          goto LABEL_19;
        }
        goto LABEL_46;
      }
      if ( v16 != *((_DWORD *)i + 27) )
      {
LABEL_18:
        v18 = 0xAAAAAAAAAAAAAAABLL * ((v20 - v15) >> 3);
        goto LABEL_19;
      }
LABEL_49:
      v31 = *((_DWORD *)v20 + 9);
      v20 += 24;
    }
    if ( v16 != v31 )
      goto LABEL_18;
LABEL_45:
    v20 += 24;
LABEL_46:
    if ( v16 != *((_DWORD *)v20 + 3) )
      goto LABEL_18;
    goto LABEL_47;
  }
LABEL_40:
  result = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v33, 0);
  if ( v41 )
    return j_j___libc_free_0(v41, v43 - (_QWORD)v41);
  return result;
}
