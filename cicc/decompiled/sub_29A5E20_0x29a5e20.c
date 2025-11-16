// Function: sub_29A5E20
// Address: 0x29a5e20
//
__int64 __fastcall sub_29A5E20(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int8 **a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 *v11; // r15
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned __int8 *v16; // r12
  __int64 (__fastcall *v17)(__int64, __int64, __int64, unsigned __int8 *, __int64, __int64, __int64, _QWORD *, __int64); // rax
  _QWORD **v18; // rdx
  int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rsi
  char *v22; // r14
  char *v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // esi
  _QWORD *v26; // rax
  __int64 v27; // r12
  unsigned __int64 v29; // r13
  __int64 v30; // rax
  unsigned int v31; // r12d
  __int64 v32; // rax
  unsigned __int8 *v33; // r14
  __int64 (__fastcall *v34)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  char *v35; // r14
  char *v36; // rbx
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // [rsp+0h] [rbp-180h]
  _QWORD *v40; // [rsp+8h] [rbp-178h]
  __int64 v41; // [rsp+10h] [rbp-170h]
  unsigned __int8 **v42; // [rsp+28h] [rbp-158h]
  unsigned __int64 v43; // [rsp+28h] [rbp-158h]
  __int64 v44; // [rsp+38h] [rbp-148h]
  unsigned __int8 **v45; // [rsp+40h] [rbp-140h] BYREF
  __int64 v46; // [rsp+48h] [rbp-138h]
  _QWORD v47[2]; // [rsp+50h] [rbp-130h] BYREF
  _BYTE v48[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v49; // [rsp+80h] [rbp-100h]
  _BYTE v50[32]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v51; // [rsp+B0h] [rbp-D0h]
  char *v52; // [rsp+C0h] [rbp-C0h] BYREF
  int v53; // [rsp+C8h] [rbp-B8h]
  char v54; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+F8h] [rbp-88h]
  __int64 v56; // [rsp+100h] [rbp-80h]
  __int64 v57; // [rsp+110h] [rbp-70h]
  __int64 v58; // [rsp+118h] [rbp-68h]
  void *v59; // [rsp+140h] [rbp-40h]

  v41 = a1;
  v40 = a3;
  v39 = a6;
  sub_23D0AB0((__int64)&v52, a1, 0, 0, 0);
  v11 = (unsigned __int8 *)v47[0];
  v45 = (unsigned __int8 **)v47;
  v46 = 0x200000000LL;
  v42 = &a4[a5];
  if ( a4 == v42 )
    goto LABEL_20;
  do
  {
    v16 = *a4;
    v49 = 257;
    v17 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, __int64, __int64, __int64, _QWORD *, __int64))(*(_QWORD *)v57 + 56LL);
    if ( (char *)v17 != (char *)sub_928890 )
    {
      v12 = v17(v57, 32, a2, v16, v9, v10, v39, v40, v41);
LABEL_5:
      if ( v12 )
        goto LABEL_6;
      goto LABEL_11;
    }
    if ( *(_BYTE *)a2 <= 0x15u && *v16 <= 0x15u )
    {
      v12 = sub_AAB310(0x20u, (unsigned __int8 *)a2, v16);
      goto LABEL_5;
    }
LABEL_11:
    v51 = 257;
    v12 = (__int64)sub_BD2C40(72, unk_3F10FD0);
    if ( v12 )
    {
      v18 = *(_QWORD ***)(a2 + 8);
      v19 = *((unsigned __int8 *)v18 + 8);
      if ( (unsigned int)(v19 - 17) > 1 )
      {
        v21 = sub_BCB2A0(*v18);
      }
      else
      {
        BYTE4(v44) = (_BYTE)v19 == 18;
        LODWORD(v44) = *((_DWORD *)v18 + 8);
        v20 = (__int64 *)sub_BCB2A0(*v18);
        v21 = sub_BCE1B0(v20, v44);
      }
      sub_B523C0(v12, v21, 53, 32, a2, (__int64)v16, (__int64)v50, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
      v58,
      v12,
      v48,
      v55,
      v56);
    v22 = v52;
    v23 = &v52[16 * v53];
    if ( v52 != v23 )
    {
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v25 = *(_DWORD *)v22;
        v22 += 16;
        sub_B99FD0(v12, v25, v24);
      }
      while ( v23 != v22 );
    }
LABEL_6:
    v13 = (unsigned int)v46;
    v14 = (unsigned int)v46 + 1LL;
    if ( v14 > HIDWORD(v46) )
    {
      sub_C8D5F0((__int64)&v45, v47, v14, 8u, v9, v10);
      v13 = (unsigned int)v46;
    }
    ++a4;
    v45[v13] = (unsigned __int8 *)v12;
    v15 = v46 + 1;
    LODWORD(v46) = v46 + 1;
  }
  while ( v42 != a4 );
  v29 = (unsigned __int64)v45;
  v43 = v15;
  v11 = *v45;
  if ( v15 > 1 )
  {
    v30 = 1;
    v31 = 1;
    do
    {
      v49 = 257;
      v33 = *(unsigned __int8 **)(v29 + 8 * v30);
      v34 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v57 + 16LL);
      if ( v34 == sub_9202E0 )
      {
        if ( *v11 > 0x15u || *v33 > 0x15u )
        {
LABEL_35:
          v51 = 257;
          v11 = (unsigned __int8 *)sub_B504D0(29, (__int64)v11, (__int64)v33, (__int64)v50, 0, 0);
          (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
            v58,
            v11,
            v48,
            v55,
            v56);
          v35 = v52;
          v36 = &v52[16 * v53];
          if ( v52 != v36 )
          {
            do
            {
              v37 = *((_QWORD *)v35 + 1);
              v38 = *(_DWORD *)v35;
              v35 += 16;
              sub_B99FD0((__int64)v11, v38, v37);
            }
            while ( v36 != v35 );
          }
          goto LABEL_32;
        }
        if ( (unsigned __int8)sub_AC47B0(29) )
          v32 = sub_AD5570(29, (__int64)v11, v33, 0, 0);
        else
          v32 = sub_AABE40(0x1Du, v11, v33);
      }
      else
      {
        v32 = v34(v57, 29u, v11, v33);
      }
      if ( !v32 )
        goto LABEL_35;
      v11 = (unsigned __int8 *)v32;
LABEL_32:
      v30 = ++v31;
    }
    while ( v31 < v43 );
  }
LABEL_20:
  v26 = sub_29A49A0(v41, (__int64)v11, v39);
  v27 = sub_29A3E20(v26, v40, 0);
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  nullsub_61();
  v59 = &unk_49DA100;
  nullsub_63();
  if ( v52 != &v54 )
    _libc_free((unsigned __int64)v52);
  return v27;
}
