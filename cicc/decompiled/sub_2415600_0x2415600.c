// Function: sub_2415600
// Address: 0x2415600
//
__int64 __fastcall sub_2415600(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4, __int16 a5, unsigned __int8 *a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r9
  __int64 v10; // r14
  __int64 (__fastcall *v11)(__int64, __int64, __int64, unsigned __int8 *, __int64); // rax
  __int64 v12; // r15
  _BYTE *v13; // r13
  _QWORD **v15; // rdx
  int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rsi
  unsigned int *v19; // rbx
  unsigned int *v20; // r14
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // [rsp-8h] [rbp-188h]
  __int64 v24; // [rsp+0h] [rbp-180h]
  unsigned __int8 *v28; // [rsp+38h] [rbp-148h]
  __int64 v30; // [rsp+48h] [rbp-138h]
  __int64 v31; // [rsp+58h] [rbp-128h]
  _BYTE v32[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v33; // [rsp+80h] [rbp-100h]
  _BYTE v34[32]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v35; // [rsp+B0h] [rbp-D0h]
  unsigned int *v36; // [rsp+C0h] [rbp-C0h] BYREF
  int v37; // [rsp+C8h] [rbp-B8h]
  char v38; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+F8h] [rbp-88h]
  __int64 v40; // [rsp+100h] [rbp-80h]
  __int64 v41; // [rsp+110h] [rbp-70h]
  __int64 v42; // [rsp+118h] [rbp-68h]
  void *v43; // [rsp+140h] [rbp-40h]

  v6 = *a3;
  v28 = a6;
  v30 = (a3[1] - *a3) >> 3;
  if ( !v30 )
    return *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  if ( !a6 )
    v28 = *(unsigned __int8 **)(*(_QWORD *)a1 + 72LL);
  v7 = 0;
  v8 = 0;
  while ( 1 )
  {
    v13 = *(_BYTE **)(v6 + 8 * v7);
    if ( *v13 <= 0x15u && sub_AC30F0(*(_QWORD *)(v6 + 8 * v7)) )
      goto LABEL_13;
    if ( !v8 )
      break;
    v10 = sub_2415280(a1, *(_QWORD *)(*a2 + 8 * v7), a4, a5);
    if ( !a4 )
      BUG();
    sub_2412230((__int64)&v36, *(_QWORD *)(a4 + 16), a4, a5, 0, v9, 0, 0);
    v33 = 257;
    v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, __int64))(*(_QWORD *)v41 + 56LL);
    if ( (char *)v11 == (char *)sub_928890 )
    {
      if ( *(_BYTE *)v10 > 0x15u || *v28 > 0x15u )
      {
LABEL_22:
        v35 = 257;
        v12 = (__int64)sub_BD2C40(72, unk_3F10FD0);
        if ( v12 )
        {
          v15 = *(_QWORD ***)(v10 + 8);
          v16 = *((unsigned __int8 *)v15 + 8);
          if ( (unsigned int)(v16 - 17) > 1 )
          {
            v18 = sub_BCB2A0(*v15);
          }
          else
          {
            BYTE4(v31) = (_BYTE)v16 == 18;
            LODWORD(v31) = *((_DWORD *)v15 + 8);
            v17 = (__int64 *)sub_BCB2A0(*v15);
            v18 = sub_BCE1B0(v17, v31);
          }
          sub_B523C0(v12, v18, 53, 33, v10, (__int64)v28, (__int64)v34, 0, 0, 0);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v42 + 16LL))(
          v42,
          v12,
          v32,
          v39,
          v40);
        if ( v36 != &v36[4 * v37] )
        {
          v24 = v7;
          v19 = v36;
          v20 = &v36[4 * v37];
          do
          {
            v21 = *((_QWORD *)v19 + 1);
            v22 = *v19;
            v19 += 4;
            sub_B99FD0(v12, v22, v21);
          }
          while ( v20 != v19 );
          v7 = v24;
        }
        goto LABEL_11;
      }
      v12 = sub_AAB310(0x21u, (unsigned __int8 *)v10, v28);
    }
    else
    {
      v12 = v11(v41, 33, v10, v28, v23);
    }
    if ( !v12 )
      goto LABEL_22;
LABEL_11:
    v35 = 257;
    v8 = sub_B36550(&v36, v12, (__int64)v13, v8, (__int64)v34, 0);
    nullsub_61();
    v43 = &unk_49DA100;
    nullsub_63();
    if ( v36 != (unsigned int *)&v38 )
      _libc_free((unsigned __int64)v36);
LABEL_13:
    if ( v30 == ++v7 )
      goto LABEL_19;
LABEL_14:
    v6 = *a3;
  }
  v8 = (__int64)v13;
  if ( v30 != ++v7 )
    goto LABEL_14;
LABEL_19:
  if ( !v8 )
    return *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  return v8;
}
