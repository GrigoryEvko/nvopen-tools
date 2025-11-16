// Function: sub_2A74E20
// Address: 0x2a74e20
//
void __fastcall sub_2A74E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, bool a5)
{
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rcx
  unsigned int v15; // eax
  unsigned int v16; // r11d
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // r12
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 *i; // rcx
  __int64 *v23; // rdi
  unsigned __int64 v24; // r15
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rcx
  int v28; // edi
  __int64 v29; // r11
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned int v36; // eax
  __int64 *v37; // rsi
  int v38; // r12d
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // r10
  __int64 v42; // rcx
  unsigned __int64 v43; // rax
  int v44; // ebx
  bool v45; // r15
  __int64 v46; // rbx
  unsigned __int8 *v47; // r12
  int v48; // eax
  int v49; // ebx
  unsigned __int64 v50; // rax
  unsigned __int8 *v51; // rax
  unsigned __int64 v52; // r15
  _BYTE *v53; // rbx
  __int64 v54; // rdx
  unsigned int v55; // esi
  int v56; // eax
  int v57; // r10d
  _BYTE v61[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v62; // [rsp+50h] [rbp-100h]
  _BYTE v63[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v64; // [rsp+80h] [rbp-D0h]
  _BYTE *v65; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+98h] [rbp-B8h]
  _BYTE v67[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+C0h] [rbp-90h]
  __int64 v69; // [rsp+C8h] [rbp-88h]
  __int64 v70; // [rsp+D0h] [rbp-80h]
  __int64 v71; // [rsp+D8h] [rbp-78h]
  void **v72; // [rsp+E0h] [rbp-70h]
  void **v73; // [rsp+E8h] [rbp-68h]
  __int64 v74; // [rsp+F0h] [rbp-60h]
  int v75; // [rsp+F8h] [rbp-58h]
  __int16 v76; // [rsp+FCh] [rbp-54h]
  char v77; // [rsp+FEh] [rbp-52h]
  __int64 v78; // [rsp+100h] [rbp-50h]
  __int64 v79; // [rsp+108h] [rbp-48h]
  void *v80; // [rsp+110h] [rbp-40h] BYREF
  void *v81; // [rsp+118h] [rbp-38h] BYREF

  if ( *(_BYTE *)a3 != 84 )
  {
    v12 = a3;
    goto LABEL_44;
  }
  v7 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a3 - 8);
    v9 = *(_QWORD *)(a1 + 40);
    v10 = 0;
    v11 = 8LL * v7;
    v12 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( a2 != *(_QWORD *)(v8 + 4 * v10) )
          goto LABEL_4;
        v13 = *(_QWORD *)(v8 + 32LL * *(unsigned int *)(a3 + 72) + v10);
        if ( v13 )
        {
          v14 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
          v15 = *(_DWORD *)(v13 + 44) + 1;
        }
        else
        {
          v14 = 0;
          v15 = 0;
        }
        v16 = *(_DWORD *)(v9 + 32);
        if ( v15 >= v16 )
          goto LABEL_4;
        v17 = *(_QWORD *)(v9 + 24);
        v18 = *(__int64 **)(v17 + 8 * v14);
        if ( !v18 )
          goto LABEL_4;
        if ( v12 )
          break;
        v50 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v50 != v13 + 48 )
        {
          if ( !v50 )
            goto LABEL_79;
          if ( (unsigned int)*(unsigned __int8 *)(v50 - 24) - 30 <= 0xA )
            v12 = v50 - 24;
        }
LABEL_4:
        v10 += 8;
        if ( v11 == v10 )
          goto LABEL_26;
      }
      v19 = *(_QWORD *)(v12 + 40);
      v20 = *(_QWORD *)(*(_QWORD *)(v19 + 72) + 80LL);
      if ( v20 )
        v20 -= 24;
      if ( v19 == v20 || v13 == v20 )
        goto LABEL_21;
      v21 = (unsigned int)(*(_DWORD *)(v19 + 44) + 1);
      if ( v16 <= (unsigned int)v21 )
        break;
      i = *(__int64 **)(v17 + 8 * v21);
      if ( v18 != i )
        goto LABEL_17;
LABEL_20:
      v20 = *v18;
LABEL_21:
      v24 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v24 == v20 + 48 )
      {
        v12 = 0;
        goto LABEL_4;
      }
      if ( !v24 )
        goto LABEL_79;
      v25 = *(unsigned __int8 *)(v24 - 24);
      v12 = v24 - 24;
      if ( (unsigned int)(v25 - 30) >= 0xB )
        v12 = 0;
      v10 += 8;
      if ( v11 == v10 )
      {
LABEL_26:
        if ( !v12 )
          return;
        if ( *(_BYTE *)a2 <= 0x1Cu )
          goto LABEL_44;
        v26 = *(_QWORD *)(a1 + 16);
        v27 = *(_QWORD *)(a2 + 40);
        v28 = *(_DWORD *)(v26 + 24);
        v29 = *(_QWORD *)(v26 + 8);
        if ( v28 )
        {
          v30 = (v28 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v31 = (__int64 *)(v29 + 16LL * v30);
          v32 = *v31;
          if ( v27 == *v31 )
          {
LABEL_30:
            v33 = v31[1];
            goto LABEL_31;
          }
          v56 = 1;
          while ( v32 != -4096 )
          {
            v57 = v56 + 1;
            v30 = (v28 - 1) & (v56 + v30);
            v31 = (__int64 *)(v29 + 16LL * v30);
            v32 = *v31;
            if ( v27 == *v31 )
              goto LABEL_30;
            v56 = v57;
          }
        }
        v33 = 0;
LABEL_31:
        v34 = *(_QWORD *)(v12 + 40);
        if ( v34 )
        {
          v35 = (unsigned int)(*(_DWORD *)(v34 + 44) + 1);
          v36 = *(_DWORD *)(v34 + 44) + 1;
        }
        else
        {
          v35 = 0;
          v36 = 0;
        }
        if ( v36 >= *(_DWORD *)(v9 + 32) || (v37 = *(__int64 **)(*(_QWORD *)(v9 + 24) + 8 * v35)) == 0 )
LABEL_79:
          BUG();
        v38 = v28 - 1;
        while ( 2 )
        {
          v42 = *v37;
          if ( v28 )
          {
            v39 = v38 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
            v40 = (__int64 *)(v29 + 16LL * v39);
            v41 = *v40;
            if ( v42 != *v40 )
            {
              v48 = 1;
              while ( v41 != -4096 )
              {
                v49 = v48 + 1;
                v39 = v38 & (v48 + v39);
                v40 = (__int64 *)(v29 + 16LL * v39);
                v41 = *v40;
                if ( v42 == *v40 )
                  goto LABEL_37;
                v48 = v49;
              }
              goto LABEL_40;
            }
LABEL_37:
            if ( v33 == v40[1] )
              goto LABEL_41;
          }
          else
          {
LABEL_40:
            if ( !v33 )
            {
LABEL_41:
              v43 = *(_QWORD *)(v42 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v43 != v42 + 48 )
              {
                if ( !v43 )
                  BUG();
                v12 = v43 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 <= 0xA )
                {
LABEL_44:
                  v44 = sub_2A73B10(a1, a2);
                  v73 = &v81;
                  v71 = sub_BD5C60(v12);
                  v72 = &v80;
                  v76 = 512;
                  v65 = v67;
                  v80 = &unk_49DA100;
                  v66 = 0x200000000LL;
                  v74 = 0;
                  v75 = 0;
                  v77 = 7;
                  v78 = 0;
                  v79 = 0;
                  v68 = 0;
                  v69 = 0;
                  LOWORD(v70) = 0;
                  v81 = &unk_49DA0B0;
                  sub_D5F1F0((__int64)&v65, v12);
                  v45 = a5;
                  if ( !a5 )
                  {
                    a5 = v44 == 1;
                    v45 = v44 == 0;
                  }
                  v46 = *(_QWORD *)(a2 + 8);
                  v62 = 257;
                  if ( v46 == *(_QWORD *)(a4 + 8) )
                  {
                    v47 = (unsigned __int8 *)a4;
                  }
                  else
                  {
                    v47 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v72 + 15))(
                                               v72,
                                               38,
                                               a4,
                                               v46);
                    if ( !v47 )
                    {
                      v64 = 257;
                      v51 = (unsigned __int8 *)sub_B51D30(38, a4, v46, (__int64)v63, 0, 0);
                      v47 = v51;
                      if ( v45 )
                        sub_B447F0(v51, 1);
                      if ( a5 )
                        sub_B44850(v47, 1);
                      (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v73 + 2))(
                        v73,
                        v47,
                        v61,
                        v69,
                        v70);
                      v52 = (unsigned __int64)v65;
                      v53 = &v65[16 * (unsigned int)v66];
                      if ( v65 != v53 )
                      {
                        do
                        {
                          v54 = *(_QWORD *)(v52 + 8);
                          v55 = *(_DWORD *)v52;
                          v52 += 16LL;
                          sub_B99FD0((__int64)v47, v55, v54);
                        }
                        while ( v53 != (_BYTE *)v52 );
                      }
                    }
                  }
                  sub_BD2ED0(a3, a2, (__int64)v47);
                  nullsub_61();
                  v80 = &unk_49DA100;
                  nullsub_63();
                  if ( v65 != v67 )
                    _libc_free((unsigned __int64)v65);
                }
              }
              return;
            }
          }
          v37 = (__int64 *)v37[1];
          if ( !v37 )
            goto LABEL_79;
          continue;
        }
      }
    }
    for ( i = 0; i != v18; i = (__int64 *)i[1] )
    {
LABEL_17:
      if ( *((_DWORD *)i + 4) < *((_DWORD *)v18 + 4) )
      {
        v23 = i;
        i = v18;
        v18 = v23;
      }
    }
    goto LABEL_20;
  }
}
