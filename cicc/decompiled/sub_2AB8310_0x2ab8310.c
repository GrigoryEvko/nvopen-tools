// Function: sub_2AB8310
// Address: 0x2ab8310
//
__int64 __fastcall sub_2AB8310(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned __int64 v4; // r14
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r12
  _BYTE *v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // r14
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  unsigned int *v19; // rbx
  unsigned int *v20; // r15
  __int64 v21; // rdx
  unsigned int v22; // esi
  _BYTE *v23; // [rsp+28h] [rbp-128h]
  _QWORD v24[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v25; // [rsp+50h] [rbp-100h]
  _QWORD v26[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+80h] [rbp-D0h]
  unsigned int *v28; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+98h] [rbp-B8h]
  _BYTE v30[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+C0h] [rbp-90h]
  __int64 v32; // [rsp+C8h] [rbp-88h]
  __int64 v33; // [rsp+D0h] [rbp-80h]
  __int64 v34; // [rsp+D8h] [rbp-78h]
  void **v35; // [rsp+E0h] [rbp-70h]
  void **v36; // [rsp+E8h] [rbp-68h]
  __int64 v37; // [rsp+F0h] [rbp-60h]
  int v38; // [rsp+F8h] [rbp-58h]
  __int16 v39; // [rsp+FCh] [rbp-54h]
  char v40; // [rsp+FEh] [rbp-52h]
  __int64 v41; // [rsp+100h] [rbp-50h]
  __int64 v42; // [rsp+108h] [rbp-48h]
  void *v43; // [rsp+110h] [rbp-40h] BYREF
  void *v44; // [rsp+118h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 360);
  v4 = sub_986580(a2);
  v34 = sub_BD5C60(v4);
  v35 = &v43;
  v36 = &v44;
  v28 = (unsigned int *)v30;
  v43 = &unk_49DA100;
  v29 = 0x200000000LL;
  v39 = 512;
  LOWORD(v33) = 0;
  v44 = &unk_49DA0B0;
  v37 = 0;
  v38 = 0;
  v40 = 7;
  v41 = 0;
  v42 = 0;
  v31 = 0;
  v32 = 0;
  sub_D5F1F0((__int64)&v28, v4);
  v5 = *(_QWORD *)(v3 + 8);
  v23 = (_BYTE *)sub_2AB26E0((__int64)&v28, v5, *(_QWORD *)(v2 + 72), *(_DWORD *)(v2 + 88));
  v6 = *(_QWORD *)(a1 + 384);
  if ( *(_BYTE *)(v6 + 108) && *(_DWORD *)(v6 + 100) )
  {
    v25 = 257;
    v26[0] = "n.rnd.up";
    v27 = 259;
    v15 = (_BYTE *)sub_AD64C0(v5, 1, 0);
    v16 = sub_929DE0(&v28, v23, v15, (__int64)v24, 0, 0);
    v17 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v35 + 4))(
                               v35,
                               13,
                               v3,
                               v16,
                               0,
                               0);
    if ( !v17 )
      v17 = sub_2AAE100((__int64 *)&v28, 13, v3, v16, (__int64)v26, 0, 0);
    v3 = (__int64)v17;
  }
  v24[0] = "n.mod.vf";
  v25 = 259;
  v7 = (*((__int64 (__fastcall **)(void **, __int64, __int64, _BYTE *))*v35 + 2))(v35, 22, v3, v23);
  if ( !v7 )
  {
    v27 = 257;
    v7 = sub_B504D0(22, v3, (__int64)v23, (__int64)v26, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v36 + 2))(v36, v7, v24, v32, v33);
    v18 = 4LL * (unsigned int)v29;
    if ( v28 != &v28[v18] )
    {
      v19 = v28;
      v20 = &v28[v18];
      do
      {
        v21 = *((_QWORD *)v19 + 1);
        v22 = *v19;
        v19 += 4;
        sub_B99FD0(v7, v22, v21);
      }
      while ( v20 != v19 );
      v2 = a1;
    }
  }
  v8 = *(_QWORD *)(v2 + 384);
  v9 = *(_DWORD *)(v2 + 72);
  if ( *(_BYTE *)(v2 + 76) && v9 )
  {
    if ( !(unsigned __int8)sub_2AB31C0(v8, 1) )
      goto LABEL_7;
  }
  else if ( !(unsigned __int8)sub_2AB31C0(v8, v9 > 1) )
  {
    goto LABEL_7;
  }
  v27 = 257;
  v13 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v7 + 8), 0, 0);
  v14 = sub_92B530(&v28, 0x20u, v7, v13, (__int64)v26);
  v27 = 257;
  v7 = sub_B36550(&v28, v14, (__int64)v23, v7, (__int64)v26, 0);
LABEL_7:
  v26[0] = "n.vec";
  v27 = 259;
  v10 = sub_929DE0(&v28, (_BYTE *)v3, (_BYTE *)v7, (__int64)v26, 0, 0);
  *(_QWORD *)(v2 + 368) = v10;
  v11 = v10;
  nullsub_61();
  v43 = &unk_49DA100;
  nullsub_63();
  if ( v28 != (unsigned int *)v30 )
    _libc_free((unsigned __int64)v28);
  return v11;
}
