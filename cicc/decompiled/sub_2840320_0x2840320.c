// Function: sub_2840320
// Address: 0x2840320
//
__int64 __fastcall sub_2840320(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // rax
  _QWORD *v10; // r12
  unsigned __int64 v11; // rax
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r14
  __int64 v14; // r14
  _QWORD **v16; // rdx
  int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r12
  _BYTE *v21; // rbx
  __int64 v22; // rdx
  unsigned int v23; // esi
  unsigned int v24; // eax
  __int64 *v25; // [rsp+10h] [rbp-170h]
  __int64 v27; // [rsp+28h] [rbp-158h]
  __int64 v28; // [rsp+58h] [rbp-128h]
  _BYTE v29[32]; // [rsp+60h] [rbp-120h] BYREF
  __int16 v30; // [rsp+80h] [rbp-100h]
  _QWORD v31[4]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v32; // [rsp+B0h] [rbp-D0h]
  _BYTE *v33; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+C8h] [rbp-B8h]
  _BYTE v35[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+F0h] [rbp-90h]
  __int64 v37; // [rsp+F8h] [rbp-88h]
  __int64 v38; // [rsp+100h] [rbp-80h]
  __int64 *v39; // [rsp+108h] [rbp-78h]
  void **v40; // [rsp+110h] [rbp-70h]
  _QWORD *v41; // [rsp+118h] [rbp-68h]
  __int64 v42; // [rsp+120h] [rbp-60h]
  int v43; // [rsp+128h] [rbp-58h]
  __int16 v44; // [rsp+12Ch] [rbp-54h]
  char v45; // [rsp+12Eh] [rbp-52h]
  __int64 v46; // [rsp+130h] [rbp-50h]
  __int64 v47; // [rsp+138h] [rbp-48h]
  void *v48; // [rsp+140h] [rbp-40h] BYREF
  _QWORD v49[7]; // [rsp+148h] [rbp-38h] BYREF

  v27 = sub_D95540(a5);
  if ( sub_DADE90(*(_QWORD *)(a1 + 16), a5, *(_QWORD *)(a1 + 40))
    && sub_DADE90(*(_QWORD *)(a1 + 16), a6, *(_QWORD *)(a1 + 40)) )
  {
    v39 = (__int64 *)sub_BD5C60(a3);
    v40 = &v48;
    v41 = v49;
    v48 = &unk_49DA100;
    v33 = v35;
    v34 = 0x200000000LL;
    v44 = 512;
    LOWORD(v38) = 0;
    v49[0] = &unk_49DA0B0;
    v42 = 0;
    v43 = 0;
    v45 = 7;
    v46 = 0;
    v47 = 0;
    v36 = 0;
    v37 = 0;
    sub_D5F1F0((__int64)&v33, a3);
    if ( (unsigned __int8)sub_DDD5B0(*(__int64 **)(a1 + 16), *(_QWORD *)(a1 + 40), a4, a5, a6) )
    {
      v14 = sub_ACD6D0(v39);
      goto LABEL_4;
    }
    v25 = *(__int64 **)(a1 + 16);
    v24 = sub_B52870(a4);
    if ( (unsigned __int8)sub_DDD5B0(v25, *(_QWORD *)(a1 + 40), v24, a5, a6) )
    {
      v14 = sub_ACD720(v39);
      goto LABEL_4;
    }
    nullsub_61();
    v48 = &unk_49DA100;
    nullsub_63();
    if ( v33 != v35 )
      _libc_free((unsigned __int64)v33);
  }
  v33 = (_BYTE *)a5;
  v9 = sub_2840260((__int64 *)a1, a2, a3, (__int64 *)&v33);
  v10 = sub_F8DB90((__int64)a2, a5, v27, v9 + 24, 0);
  v33 = (_BYTE *)a6;
  v11 = sub_2840260((__int64 *)a1, a2, a3, (__int64 *)&v33);
  v12 = sub_F8DB90((__int64)a2, a6, v27, v11 + 24, 0);
  v31[0] = v10;
  v31[1] = v12;
  v13 = sub_28401D0(a1, a3, (__int64)v31, 2);
  v39 = (__int64 *)sub_BD5C60(v13);
  v33 = v35;
  v40 = &v48;
  v34 = 0x200000000LL;
  v41 = v49;
  v44 = 512;
  LOWORD(v38) = 0;
  v48 = &unk_49DA100;
  v42 = 0;
  v43 = 0;
  v49[0] = &unk_49DA0B0;
  v45 = 7;
  v46 = 0;
  v47 = 0;
  v36 = 0;
  v37 = 0;
  sub_D5F1F0((__int64)&v33, v13);
  v30 = 257;
  v14 = (*((__int64 (__fastcall **)(void **, _QWORD, _QWORD *, _QWORD *))*v40 + 7))(v40, a4, v10, v12);
  if ( !v14 )
  {
    v32 = 257;
    v14 = (__int64)sub_BD2C40(72, unk_3F10FD0);
    if ( v14 )
    {
      v16 = (_QWORD **)v10[1];
      v17 = *((unsigned __int8 *)v16 + 8);
      if ( (unsigned int)(v17 - 17) > 1 )
      {
        v19 = sub_BCB2A0(*v16);
      }
      else
      {
        BYTE4(v28) = (_BYTE)v17 == 18;
        LODWORD(v28) = *((_DWORD *)v16 + 8);
        v18 = (__int64 *)sub_BCB2A0(*v16);
        v19 = sub_BCE1B0(v18, v28);
      }
      sub_B523C0(v14, v19, 53, a4, (__int64)v10, (__int64)v12, (__int64)v31, 0, 0, 0);
    }
    (*(void (__fastcall **)(_QWORD *, __int64, _BYTE *, __int64, __int64))(*v41 + 16LL))(v41, v14, v29, v37, v38);
    v20 = (__int64)v33;
    v21 = &v33[16 * (unsigned int)v34];
    if ( v33 != v21 )
    {
      do
      {
        v22 = *(_QWORD *)(v20 + 8);
        v23 = *(_DWORD *)v20;
        v20 += 16;
        sub_B99FD0(v14, v23, v22);
      }
      while ( v21 != (_BYTE *)v20 );
    }
  }
LABEL_4:
  nullsub_61();
  v48 = &unk_49DA100;
  nullsub_63();
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  return v14;
}
