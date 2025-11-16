// Function: sub_2A749F0
// Address: 0x2a749f0
//
__int64 __fastcall sub_2A749F0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  int v9; // ecx
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  _QWORD *v15; // r14
  int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r13
  int v24; // r12d
  unsigned __int64 v25; // r12
  _BYTE *v26; // rbx
  __int64 v27; // rdx
  unsigned int v28; // esi
  _QWORD *v30; // rax
  unsigned __int64 v31; // r12
  _BYTE *v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // esi
  int v35; // eax
  int v36; // r9d
  _BYTE v39[32]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v40; // [rsp+40h] [rbp-100h]
  _BYTE v41[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v42; // [rsp+70h] [rbp-D0h]
  _BYTE *v43; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+88h] [rbp-B8h]
  _BYTE v45[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+B0h] [rbp-90h]
  __int64 v47; // [rsp+B8h] [rbp-88h]
  __int64 v48; // [rsp+C0h] [rbp-80h]
  __int64 v49; // [rsp+C8h] [rbp-78h]
  void **v50; // [rsp+D0h] [rbp-70h]
  void **v51; // [rsp+D8h] [rbp-68h]
  __int64 v52; // [rsp+E0h] [rbp-60h]
  int v53; // [rsp+E8h] [rbp-58h]
  __int16 v54; // [rsp+ECh] [rbp-54h]
  char v55; // [rsp+EEh] [rbp-52h]
  __int64 v56; // [rsp+F0h] [rbp-50h]
  __int64 v57; // [rsp+F8h] [rbp-48h]
  void *v58; // [rsp+100h] [rbp-40h] BYREF
  void *v59; // [rsp+108h] [rbp-38h] BYREF

  v49 = sub_BD5C60(a5);
  v50 = &v58;
  v43 = v45;
  v58 = &unk_49DA100;
  v54 = 512;
  v44 = 0x200000000LL;
  v59 = &unk_49DA0B0;
  v51 = &v59;
  v52 = 0;
  v53 = 0;
  v55 = 7;
  v56 = 0;
  v57 = 0;
  v46 = 0;
  v47 = 0;
  LOWORD(v48) = 0;
  sub_D5F1F0((__int64)&v43, a5);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_QWORD *)(a5 + 40);
  v9 = *(_DWORD *)(v7 + 24);
  v10 = *(_QWORD *)(v7 + 8);
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_3:
      v15 = (_QWORD *)v13[1];
      if ( v15 )
      {
        while ( sub_D4B130((__int64)v15) && (unsigned __int8)sub_D48480((__int64)v15, a2, v18, v19) )
        {
          v20 = sub_D4B130((__int64)v15);
          v21 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v21 == v20 + 48 )
          {
            sub_D5F1F0((__int64)&v43, 0);
            v15 = (_QWORD *)*v15;
            if ( !v15 )
              break;
          }
          else
          {
            if ( !v21 )
              BUG();
            v16 = *(unsigned __int8 *)(v21 - 24);
            v17 = v21 - 24;
            if ( (unsigned int)(v16 - 30) >= 0xB )
              v17 = 0;
            sub_D5F1F0((__int64)&v43, v17);
            v15 = (_QWORD *)*v15;
            if ( !v15 )
              break;
          }
        }
      }
    }
    else
    {
      v35 = 1;
      while ( v14 != -4096 )
      {
        v36 = v35 + 1;
        v12 = v11 & (v35 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          goto LABEL_3;
        v35 = v36;
      }
    }
  }
  v22 = *(_QWORD *)(a2 + 8);
  if ( a4 )
  {
    v40 = 257;
    if ( v22 != a3 )
    {
      v23 = (*((__int64 (__fastcall **)(void **, __int64, __int64))*v50 + 15))(v50, 40, a2);
      if ( !v23 )
      {
        v42 = 257;
        v23 = sub_B51D30(40, a2, a3, (__int64)v41, 0, 0);
        if ( (unsigned __int8)sub_920620(v23) )
        {
          v24 = v53;
          if ( v52 )
            sub_B99FD0(v23, 3u, v52);
          sub_B45150(v23, v24);
        }
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v51 + 2))(v51, v23, v39, v47, v48);
        v25 = (unsigned __int64)v43;
        v26 = &v43[16 * (unsigned int)v44];
        if ( v43 != v26 )
        {
          do
          {
            v27 = *(_QWORD *)(v25 + 8);
            v28 = *(_DWORD *)v25;
            v25 += 16LL;
            sub_B99FD0(v23, v28, v27);
          }
          while ( v26 != (_BYTE *)v25 );
        }
      }
      goto LABEL_25;
    }
LABEL_28:
    v23 = a2;
    goto LABEL_25;
  }
  v40 = 257;
  if ( v22 == a3 )
    goto LABEL_28;
  v23 = (*((__int64 (__fastcall **)(void **, __int64, __int64))*v50 + 15))(v50, 39, a2);
  if ( !v23 )
  {
    v42 = 257;
    v30 = sub_BD2C40(72, 1u);
    v23 = (__int64)v30;
    if ( v30 )
      sub_B515B0((__int64)v30, a2, a3, (__int64)v41, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v51 + 2))(v51, v23, v39, v47, v48);
    v31 = (unsigned __int64)v43;
    v32 = &v43[16 * (unsigned int)v44];
    if ( v43 != v32 )
    {
      do
      {
        v33 = *(_QWORD *)(v31 + 8);
        v34 = *(_DWORD *)v31;
        v31 += 16LL;
        sub_B99FD0(v23, v34, v33);
      }
      while ( v32 != (_BYTE *)v31 );
    }
  }
LABEL_25:
  nullsub_61();
  v58 = &unk_49DA100;
  nullsub_63();
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  return v23;
}
