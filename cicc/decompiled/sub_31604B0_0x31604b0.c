// Function: sub_31604B0
// Address: 0x31604b0
//
__int64 __fastcall sub_31604B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 *v5; // rax
  __int64 v6; // rax
  char v7; // bl
  _QWORD *v8; // rax
  __int64 v9; // r9
  __int64 v10; // r15
  unsigned int *v11; // r13
  unsigned int *v12; // rbx
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // rbx
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r13
  unsigned int *v22; // rax
  int v23; // ecx
  unsigned int *v24; // rdx
  unsigned __int64 v25; // r13
  __int64 **v26; // rax
  int v27; // eax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v33; // rsi
  __int64 v34; // rsi
  unsigned __int64 v35; // rsi
  __int64 v36; // [rsp-10h] [rbp-170h]
  __int64 *v38; // [rsp+28h] [rbp-138h]
  _BYTE v39[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v40; // [rsp+60h] [rbp-100h]
  _QWORD v41[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v42; // [rsp+90h] [rbp-D0h]
  unsigned int *v43; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+A8h] [rbp-B8h]
  _BYTE v45[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+D0h] [rbp-90h]
  __int64 v47; // [rsp+D8h] [rbp-88h]
  __int64 v48; // [rsp+E0h] [rbp-80h]
  __int64 *v49; // [rsp+E8h] [rbp-78h]
  void **v50; // [rsp+F0h] [rbp-70h]
  void **v51; // [rsp+F8h] [rbp-68h]
  __int64 v52; // [rsp+100h] [rbp-60h]
  int v53; // [rsp+108h] [rbp-58h]
  __int16 v54; // [rsp+10Ch] [rbp-54h]
  char v55; // [rsp+10Eh] [rbp-52h]
  __int64 v56; // [rsp+110h] [rbp-50h]
  __int64 v57; // [rsp+118h] [rbp-48h]
  void *v58; // [rsp+120h] [rbp-40h] BYREF
  void *v59; // [rsp+128h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 + 72);
  v5 = (__int64 *)sub_BD5C60(a1);
  v55 = 7;
  v49 = v5;
  v50 = &v58;
  v51 = &v59;
  LOWORD(v48) = 0;
  v43 = (unsigned int *)v45;
  v58 = &unk_49DA100;
  v44 = 0x200000000LL;
  v54 = 512;
  v59 = &unk_49DA0B0;
  v52 = 0;
  v53 = 0;
  v56 = 0;
  v57 = 0;
  v46 = 0;
  v47 = 0;
  sub_D5F1F0((__int64)&v43, a1);
  v40 = 257;
  v6 = sub_AA4E30(v46);
  v38 = (__int64 *)v4;
  v7 = sub_AE5020(v6, v4);
  v42 = 257;
  v8 = sub_BD2C40(80, 1u);
  v10 = (__int64)v8;
  if ( v8 )
  {
    sub_B4D190((__int64)v8, (__int64)v38, a2, (__int64)v41, 0, v7, 0, 0);
    v9 = v36;
  }
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64, __int64))*v51 + 2))(v51, v10, v39, v47, v48, v9);
  v11 = v43;
  v12 = &v43[4 * (unsigned int)v44];
  if ( v43 != v12 )
  {
    do
    {
      v13 = *((_QWORD *)v11 + 1);
      v14 = *v11;
      v11 += 4;
      sub_B99FD0(v10, v14, v13);
    }
    while ( v12 != v11 );
  }
  v15 = sub_3160280((__int64)&v43, v10, a3);
  if ( *(_BYTE *)a1 == 85 )
  {
    v33 = *(_QWORD *)(a1 + 32);
    if ( v33 == *(_QWORD *)(a1 + 40) + 48LL || !v33 )
      v34 = 0;
    else
      v34 = v33 - 24;
    sub_D5F1F0((__int64)&v43, v34);
  }
  else
  {
    v16 = sub_AA5030(*(_QWORD *)(a1 - 96), 1);
    if ( !v16 )
      BUG();
    v46 = *(_QWORD *)(v16 + 16);
    v47 = v16;
    LOWORD(v48) = v17;
    v18 = *(_QWORD *)sub_B46C60(v16 - 24);
    v41[0] = v18;
    if ( v18 && (sub_B96E90((__int64)v41, v18, 1), (v21 = v41[0]) != 0) )
    {
      v22 = v43;
      v23 = v44;
      v24 = &v43[4 * (unsigned int)v44];
      if ( v43 != v24 )
      {
        while ( 1 )
        {
          v19 = *v22;
          if ( !(_DWORD)v19 )
            break;
          v22 += 4;
          if ( v24 == v22 )
            goto LABEL_27;
        }
        *((_QWORD *)v22 + 1) = v41[0];
        goto LABEL_14;
      }
LABEL_27:
      if ( (unsigned int)v44 >= (unsigned __int64)HIDWORD(v44) )
      {
        v35 = (unsigned int)v44 + 1LL;
        if ( HIDWORD(v44) < v35 )
        {
          sub_C8D5F0((__int64)&v43, v45, v35, 0x10u, v19, v20);
          v24 = &v43[4 * (unsigned int)v44];
        }
        *(_QWORD *)v24 = 0;
        *((_QWORD *)v24 + 1) = v21;
        v21 = v41[0];
        LODWORD(v44) = v44 + 1;
      }
      else
      {
        if ( v24 )
        {
          *v24 = 0;
          *((_QWORD *)v24 + 1) = v21;
          v23 = v44;
          v21 = v41[0];
        }
        LODWORD(v44) = v23 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v43, 0);
      v21 = v41[0];
    }
    if ( v21 )
LABEL_14:
      sub_B91220((__int64)v41, v21);
  }
  v25 = sub_BCF480(v38, 0, 0, 0);
  v26 = (__int64 **)sub_BCE3C0(v49, 0);
  v27 = sub_AC9EC0(v26);
  v42 = 257;
  v30 = sub_921880(&v43, v25, v27, 0, 0, (__int64)v41, 0);
  v31 = *(unsigned int *)(a3 + 256);
  if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 260) )
  {
    sub_C8D5F0(a3 + 248, (const void *)(a3 + 264), v31 + 1, 8u, v28, v29);
    v31 = *(unsigned int *)(a3 + 256);
  }
  *(_QWORD *)(*(_QWORD *)(a3 + 248) + 8 * v31) = v30;
  ++*(_DWORD *)(a3 + 256);
  sub_315E620((__int64 *)&v43, v30, a2, 0, 0);
  nullsub_61();
  v58 = &unk_49DA100;
  nullsub_63();
  if ( v43 != (unsigned int *)v45 )
    _libc_free((unsigned __int64)v43);
  return v15;
}
