// Function: sub_24161C0
// Address: 0x24161c0
//
void __fastcall sub_24161C0(__int64 *a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // r15
  __int64 v5; // rax
  int v6; // eax
  unsigned __int8 *v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  __int64 **v10; // r14
  __int64 v11; // r15
  unsigned int v12; // ebx
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rbx
  unsigned int *v19; // r14
  unsigned int *v20; // r15
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 (__fastcall *v23)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned int *v24; // rbx
  unsigned int *v25; // r14
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // [rsp+0h] [rbp-180h]
  __int64 v29; // [rsp+8h] [rbp-178h]
  _QWORD v30[4]; // [rsp+10h] [rbp-170h] BYREF
  _BYTE v31[32]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v32; // [rsp+50h] [rbp-130h]
  char v33[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v34; // [rsp+80h] [rbp-100h]
  _BYTE v35[32]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v36; // [rsp+B0h] [rbp-D0h]
  unsigned int *v37; // [rsp+C0h] [rbp-C0h] BYREF
  int v38; // [rsp+C8h] [rbp-B8h]
  char v39; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+F8h] [rbp-88h]
  __int64 v41; // [rsp+100h] [rbp-80h]
  __int64 v42; // [rsp+110h] [rbp-70h]
  __int64 v43; // [rsp+118h] [rbp-68h]
  void *v44; // [rsp+140h] [rbp-40h]

  sub_23D0AB0((__int64)&v37, a2, 0, 0, 0);
  v3 = *(_BYTE **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v4 = sub_24159D0(*a1, (__int64)v3);
  if ( (unsigned __int8)sub_240D530() )
  {
    v3 = *(_BYTE **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v5 = sub_2414930((__int64 *)*a1, v3);
  }
  else
  {
    v5 = *(_QWORD *)(*(_QWORD *)*a1 + 40LL);
  }
  v30[1] = v5;
  v6 = *(_DWORD *)(a2 + 4);
  v34 = 257;
  v30[0] = v4;
  v7 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (v6 & 0x7FFFFFF)), (__int64)v3);
  v8 = *a1;
  v9 = *(_DWORD *)(a2 + 4);
  v32 = 257;
  v30[2] = v7;
  v28 = v8;
  v10 = *(__int64 ***)(*(_QWORD *)v8 + 64LL);
  v11 = *(_QWORD *)(a2 + 32 * (2LL - (v9 & 0x7FFFFFF)));
  v29 = *(_QWORD *)(v11 + 8);
  v12 = sub_BCB060(v29);
  v13 = sub_BCB060((__int64)v10);
  v14 = v28;
  if ( v12 >= v13 )
  {
    if ( v10 == (__int64 **)v29 || v12 == v13 )
      goto LABEL_6;
    v23 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v42 + 120LL);
    if ( v23 == sub_920130 )
    {
      if ( *(_BYTE *)v11 > 0x15u )
      {
LABEL_28:
        v36 = 257;
        v11 = sub_B51D30(38, v11, (__int64)v10, (__int64)v35, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v43 + 16LL))(
          v43,
          v11,
          v31,
          v40,
          v41);
        v24 = v37;
        v25 = &v37[4 * v38];
        if ( v37 != v25 )
        {
          do
          {
            v26 = *((_QWORD *)v24 + 1);
            v27 = *v24;
            v24 += 4;
            sub_B99FD0(v11, v27, v26);
          }
          while ( v25 != v24 );
        }
        v14 = *a1;
        goto LABEL_6;
      }
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v16 = sub_ADAB70(38, v11, v10, 0);
      else
        v16 = sub_AA93C0(0x26u, v11, (__int64)v10);
    }
    else
    {
      v16 = v23(v42, 38u, (_BYTE *)v11, (__int64)v10);
    }
    if ( v16 )
      goto LABEL_26;
    goto LABEL_28;
  }
  if ( v10 == (__int64 **)v29 )
    goto LABEL_6;
  v15 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v42 + 120LL);
  if ( v15 != sub_920130 )
  {
    v16 = v15(v42, 39u, (_BYTE *)v11, (__int64)v10);
    goto LABEL_15;
  }
  if ( *(_BYTE *)v11 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v16 = sub_ADAB70(39, v11, v10, 0);
    else
      v16 = sub_AA93C0(0x27u, v11, (__int64)v10);
LABEL_15:
    if ( v16 )
    {
LABEL_26:
      v14 = *a1;
      v11 = v16;
      goto LABEL_6;
    }
  }
  v36 = 257;
  v17 = sub_BD2C40(72, unk_3F10A14);
  v18 = (__int64)v17;
  if ( v17 )
    sub_B515B0((__int64)v17, v11, (__int64)v10, (__int64)v35, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v43 + 16LL))(v43, v18, v31, v40, v41);
  v19 = v37;
  v20 = &v37[4 * v38];
  if ( v37 != v20 )
  {
    do
    {
      v21 = *((_QWORD *)v19 + 1);
      v22 = *v19;
      v19 += 4;
      sub_B99FD0(v18, v22, v21);
    }
    while ( v20 != v19 );
  }
  v14 = *a1;
  v11 = v18;
LABEL_6:
  v30[3] = v11;
  sub_921880(
    &v37,
    *(_QWORD *)(*(_QWORD *)v14 + 344LL),
    *(_QWORD *)(*(_QWORD *)v14 + 352LL),
    (int)v30,
    4,
    (__int64)v33,
    0);
  nullsub_61();
  v44 = &unk_49DA100;
  nullsub_63();
  if ( v37 != (unsigned int *)&v39 )
    _libc_free((unsigned __int64)v37);
}
