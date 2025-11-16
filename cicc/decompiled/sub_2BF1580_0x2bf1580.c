// Function: sub_2BF1580
// Address: 0x2bf1580
//
void __fastcall sub_2BF1580(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rax
  int *v13; // rax
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // r11
  __int64 v17; // rax
  unsigned __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rsi
  _BYTE *v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r15
  unsigned int *v26; // r15
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v30; // [rsp+20h] [rbp-130h]
  unsigned int *v31; // [rsp+20h] [rbp-130h]
  unsigned int v32; // [rsp+28h] [rbp-128h]
  __int64 v33; // [rsp+28h] [rbp-128h]
  __int64 v34; // [rsp+28h] [rbp-128h]
  __int64 v35; // [rsp+28h] [rbp-128h]
  _BYTE v36[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v37; // [rsp+50h] [rbp-100h]
  _QWORD v38[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v39; // [rsp+80h] [rbp-D0h]
  unsigned int *v40; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+98h] [rbp-B8h]
  _BYTE v42[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+C0h] [rbp-90h]
  __int64 v44; // [rsp+C8h] [rbp-88h]
  __int64 v45; // [rsp+D0h] [rbp-80h]
  __int64 v46; // [rsp+D8h] [rbp-78h]
  void **v47; // [rsp+E0h] [rbp-70h]
  _QWORD *v48; // [rsp+E8h] [rbp-68h]
  __int64 v49; // [rsp+F0h] [rbp-60h]
  int v50; // [rsp+F8h] [rbp-58h]
  __int16 v51; // [rsp+FCh] [rbp-54h]
  char v52; // [rsp+FEh] [rbp-52h]
  __int64 v53; // [rsp+100h] [rbp-50h]
  __int64 v54; // [rsp+108h] [rbp-48h]
  void *v55; // [rsp+110h] [rbp-40h] BYREF
  _QWORD v56[7]; // [rsp+118h] [rbp-38h] BYREF

  v4 = a4;
  v6 = *(_QWORD *)(a1 + 208);
  v7 = *(_QWORD *)(a2 + 8);
  if ( v6 && *(_DWORD *)(v6 + 24) )
  {
    v17 = *(_QWORD *)(a4 + 104);
    v19 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v19 == v17 + 48 )
    {
      v21 = 0;
    }
    else
    {
      if ( !v19 )
        BUG();
      v20 = *(unsigned __int8 *)(v19 - 24);
      v21 = v19 - 24;
      if ( (unsigned int)(v20 - 30) >= 0xB )
        v21 = 0;
    }
    v46 = sub_BD5C60(v21);
    v48 = v56;
    v41 = 0x200000000LL;
    v47 = &v55;
    v40 = (unsigned int *)v42;
    v51 = 512;
    LOWORD(v45) = 0;
    v55 = &unk_49DA100;
    v56[0] = &unk_49DA0B0;
    v49 = 0;
    v50 = 0;
    v52 = 7;
    v53 = 0;
    v54 = 0;
    v43 = 0;
    v44 = 0;
    sub_D5F1F0((__int64)&v40, v21);
    v38[0] = "trip.count.minus.1";
    v39 = 259;
    v22 = (_BYTE *)sub_AD64C0(v7, 1, 0);
    *(_QWORD *)(*(_QWORD *)(a1 + 208) + 40LL) = sub_929DE0(&v40, (_BYTE *)a2, v22, (__int64)v38, 0, 0);
    nullsub_61();
    v55 = &unk_49DA100;
    nullsub_63();
    v4 = a4;
    if ( v40 != (unsigned int *)v42 )
    {
      _libc_free((unsigned __int64)v40);
      v4 = a4;
    }
  }
  *(_QWORD *)(a1 + 256) = a3;
  v8 = *(_QWORD *)(v4 + 104);
  v9 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v8 + 48 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = v9 - 24;
    if ( (unsigned int)(v10 - 30) >= 0xB )
      v11 = 0;
  }
  v30 = v4;
  v12 = sub_BD5C60(v11);
  v47 = &v55;
  v46 = v12;
  LOWORD(v45) = 0;
  v51 = 512;
  v55 = &unk_49DA100;
  v40 = (unsigned int *)v42;
  v56[0] = &unk_49DA0B0;
  v41 = 0x200000000LL;
  v48 = v56;
  v49 = 0;
  v50 = 0;
  v52 = 7;
  v53 = 0;
  v54 = 0;
  v43 = 0;
  v44 = 0;
  sub_D5F1F0((__int64)&v40, v11);
  v13 = *(int **)(a1 + 144);
  if ( *(_DWORD *)(a1 + 296) )
  {
    v32 = *v13;
    v14 = sub_2AB2710((__int64)&v40, v7, *(_QWORD *)(v30 + 8));
    v15 = v32;
    *(_QWORD *)(a1 + 312) = v14;
    v16 = v14;
    if ( v32 > 1 )
    {
      v33 = v14;
      v37 = 257;
      v23 = sub_AD64C0(v7, v15, 0);
      v24 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v47 + 4))(
              v47,
              17,
              v33,
              v23,
              0,
              0);
      if ( v24 )
      {
        v16 = v24;
      }
      else
      {
        v39 = 257;
        v34 = sub_B504D0(17, v33, v23, (__int64)v38, 0, 0);
        (*(void (__fastcall **)(_QWORD *, __int64, _BYTE *, __int64, __int64))(*v48 + 16LL))(v48, v34, v36, v44, v45);
        v16 = v34;
        v25 = 4LL * (unsigned int)v41;
        v31 = &v40[v25];
        if ( v40 != &v40[v25] )
        {
          v26 = v40;
          do
          {
            v27 = *((_QWORD *)v26 + 1);
            v28 = *v26;
            v26 += 4;
            v35 = v16;
            sub_B99FD0(v16, v28, v27);
            v16 = v35;
          }
          while ( v31 != v26 );
        }
      }
    }
    *(_QWORD *)(a1 + 368) = v16;
  }
  else
  {
    *(_QWORD *)(a1 + 368) = sub_2AB26E0((__int64)&v40, v7, *(_QWORD *)(v30 + 8), *v13);
  }
  nullsub_61();
  v55 = &unk_49DA100;
  nullsub_63();
  if ( v40 != (unsigned int *)v42 )
    _libc_free((unsigned __int64)v40);
}
