// Function: sub_F8D4E0
// Address: 0xf8d4e0
//
__int64 __fastcall sub_F8D4E0(__int64 a1, __int64 a2, unsigned int a3, char a4, __int64 a5, __int64 a6, char a7)
{
  __int64 v8; // rbx
  __int64 v9; // r13
  bool v10; // sf
  int v11; // eax
  __int64 v12; // r14
  _QWORD *v13; // rax
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rax
  unsigned int v20; // r14d
  _QWORD *v21; // r15
  __int64 v23; // rdx
  int v24; // ecx
  int v25; // eax
  _QWORD *v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // rdx
  unsigned int v34; // esi
  _QWORD *v35; // rax
  __int64 v36; // r12
  __int64 v37; // r14
  __int64 v38; // r13
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // [rsp+0h] [rbp-E0h]
  __int64 v43; // [rsp+18h] [rbp-C8h]
  __int64 v44; // [rsp+20h] [rbp-C0h]
  char v45; // [rsp+2Ah] [rbp-B6h]
  __int64 v48; // [rsp+30h] [rbp-B0h]
  __int64 v49; // [rsp+38h] [rbp-A8h]
  int v50; // [rsp+38h] [rbp-A8h]
  __int64 v51; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v52; // [rsp+48h] [rbp-98h]
  __int64 v53[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v54; // [rsp+70h] [rbp-70h]
  _QWORD v55[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v56; // [rsp+A0h] [rbp-40h]

  v8 = a1;
  v45 = *(_BYTE *)(a1 + 514);
  *(_BYTE *)(a1 + 514) = a4 | v45;
  v9 = sub_F894B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a2 + 40) - 1)));
  v43 = *(_QWORD *)(v9 + 8);
  if ( a4 )
  {
    v54 = 257;
    v56 = 257;
    v35 = sub_BD2C40(72, unk_3F10A14);
    v36 = (__int64)v35;
    if ( v35 )
      sub_B549F0((__int64)v35, v9, (__int64)v55, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
      *(_QWORD *)(a1 + 608),
      v36,
      v53,
      *(_QWORD *)(a1 + 576),
      *(_QWORD *)(a1 + 584));
    v37 = *(_QWORD *)(a1 + 520);
    v38 = v37 + 16LL * *(unsigned int *)(a1 + 528);
    while ( v38 != v37 )
    {
      v39 = *(_QWORD *)(v37 + 8);
      v40 = *(_DWORD *)v37;
      v37 += 16;
      sub_B99FD0(v36, v40, v39);
    }
    v9 = v36;
  }
  v49 = *(_QWORD *)(a2 + 40);
  v10 = (int)v49 - 2 < 0;
  v11 = v49 - 2;
  v50 = v49 - 2;
  if ( !v10 )
  {
    v44 = a1 + 520;
    v48 = 8LL * v11;
    do
    {
      if ( v50 && a4 )
      {
        *(_BYTE *)(v8 + 514) = 1;
        v12 = sub_F894B0(v8, *(_QWORD *)(*(_QWORD *)(a2 + 32) + v48));
        v56 = 257;
        v54 = 257;
        v13 = sub_BD2C40(72, unk_3F10A14);
        v14 = (__int64)v13;
        if ( v13 )
          sub_B549F0((__int64)v13, v12, (__int64)v55, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 608) + 16LL))(
          *(_QWORD *)(v8 + 608),
          v14,
          v53,
          *(_QWORD *)(v44 + 56),
          *(_QWORD *)(v44 + 64));
        v15 = *(_QWORD *)(v8 + 520);
        v16 = v15 + 16LL * *(unsigned int *)(v8 + 528);
        while ( v16 != v15 )
        {
          v17 = *(_QWORD *)(v15 + 8);
          v18 = *(_DWORD *)v15;
          v15 += 16;
          sub_B99FD0(v14, v18, v17);
        }
        v19 = v43;
        if ( *(_BYTE *)(v43 + 8) == 12 )
        {
LABEL_24:
          v51 = v19;
          BYTE4(v53[0]) = 0;
          v55[0] = v9;
          v55[1] = v14;
          v9 = sub_B33D10(v44, a3, (__int64)&v51, 1, (int)v55, 2, v53[0], (__int64)&a7);
          goto LABEL_17;
        }
      }
      else
      {
        *(_BYTE *)(v8 + 514) = v45;
        v14 = sub_F894B0(v8, *(_QWORD *)(*(_QWORD *)(a2 + 32) + v48));
        v19 = v43;
        if ( *(_BYTE *)(v43 + 8) == 12 )
          goto LABEL_24;
      }
      v54 = 257;
      if ( a3 == 365 )
      {
        v20 = 34;
      }
      else if ( a3 > 0x16D )
      {
        if ( a3 != 366 )
LABEL_40:
          BUG();
        v20 = 36;
      }
      else if ( a3 == 329 )
      {
        v20 = 38;
      }
      else
      {
        if ( a3 != 330 )
          goto LABEL_40;
        v20 = 40;
      }
      v21 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v8 + 600) + 56LL))(
                        *(_QWORD *)(v8 + 600),
                        v20,
                        v9,
                        v14);
      if ( !v21 )
      {
        v56 = 257;
        v21 = sub_BD2C40(72, unk_3F10FD0);
        if ( v21 )
        {
          v23 = *(_QWORD *)(v9 + 8);
          v24 = *(unsigned __int8 *)(v23 + 8);
          if ( (unsigned int)(v24 - 17) > 1 )
          {
            v28 = sub_BCB2A0(*(_QWORD **)v23);
          }
          else
          {
            v25 = *(_DWORD *)(v23 + 32);
            v26 = *(_QWORD **)v23;
            BYTE4(v52) = (_BYTE)v24 == 18;
            LODWORD(v52) = v25;
            v27 = (__int64 *)sub_BCB2A0(v26);
            v28 = sub_BCE1B0(v27, v52);
          }
          sub_B523C0((__int64)v21, v28, 53, v20, v9, v14, (__int64)v55, 0, 0, 0);
        }
        (*(void (__fastcall **)(_QWORD, _QWORD *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 608) + 16LL))(
          *(_QWORD *)(v8 + 608),
          v21,
          v53,
          *(_QWORD *)(v44 + 56),
          *(_QWORD *)(v44 + 64));
        v29 = *(_QWORD *)(v8 + 520);
        v30 = v29 + 16LL * *(unsigned int *)(v8 + 528);
        if ( v29 != v30 )
        {
          v41 = v8;
          v31 = *(_QWORD *)(v8 + 520);
          v32 = v30;
          do
          {
            v33 = *(_QWORD *)(v31 + 8);
            v34 = *(_DWORD *)v31;
            v31 += 16;
            sub_B99FD0((__int64)v21, v34, v33);
          }
          while ( v32 != v31 );
          v8 = v41;
        }
      }
      v9 = sub_B36550((unsigned int **)v44, (__int64)v21, v9, v14, (__int64)&a7, 0);
LABEL_17:
      --v50;
      v48 -= 8;
    }
    while ( v50 != -1 );
  }
  *(_BYTE *)(v8 + 514) = v45;
  return v9;
}
