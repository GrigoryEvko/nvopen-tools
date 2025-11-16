// Function: sub_2D5E810
// Address: 0x2d5e810
//
__int64 __fastcall sub_2D5E810(__int64 a1, unsigned __int64 a2, __int64 **a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r9
  _QWORD *v6; // r12
  __int64 (__fastcall *v7)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  char *v11; // rcx
  __int64 v12; // r15
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // r8
  int v15; // edx
  _QWORD *v16; // rax
  unsigned __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  char *v21; // r13
  _BYTE *v23; // [rsp+18h] [rbp-128h]
  char *v24; // [rsp+20h] [rbp-120h] BYREF
  char v25; // [rsp+40h] [rbp-100h]
  char v26; // [rsp+41h] [rbp-FFh]
  _QWORD v27[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v28; // [rsp+70h] [rbp-D0h]
  _BYTE *v29; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+88h] [rbp-B8h]
  _BYTE v31[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v32; // [rsp+B0h] [rbp-90h]
  __int64 v33; // [rsp+B8h] [rbp-88h]
  __int64 v34; // [rsp+C0h] [rbp-80h]
  __int64 v35; // [rsp+C8h] [rbp-78h]
  void **v36; // [rsp+D0h] [rbp-70h]
  void **v37; // [rsp+D8h] [rbp-68h]
  __int64 v38; // [rsp+E0h] [rbp-60h]
  int v39; // [rsp+E8h] [rbp-58h]
  __int16 v40; // [rsp+ECh] [rbp-54h]
  char v41; // [rsp+EEh] [rbp-52h]
  __int64 v42; // [rsp+F0h] [rbp-50h]
  __int64 v43; // [rsp+F8h] [rbp-48h]
  void *v44; // [rsp+100h] [rbp-40h] BYREF
  void *v45; // [rsp+108h] [rbp-38h] BYREF

  v4 = (_QWORD *)sub_22077B0(0x18u);
  v6 = v4;
  if ( !v4 )
    goto LABEL_10;
  v4[1] = a2;
  *v4 = off_49D4090;
  v35 = sub_BD5C60(a2);
  v36 = &v44;
  v37 = &v45;
  v29 = v31;
  v44 = &unk_49DA100;
  v30 = 0x200000000LL;
  v40 = 512;
  LOWORD(v34) = 0;
  v45 = &unk_49DA0B0;
  v38 = 0;
  v39 = 0;
  v41 = 7;
  v42 = 0;
  v43 = 0;
  v32 = 0;
  v33 = 0;
  sub_D5F1F0((__int64)&v29, a2);
  v27[0] = 0;
  sub_93FB40((__int64)&v29, 0);
  v26 = 1;
  v24 = "promoted";
  v25 = 3;
  if ( a3 != *(__int64 ***)(a2 + 8) )
  {
    v7 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v36 + 15);
    if ( v7 == sub_920130 )
    {
      if ( *(_BYTE *)a2 > 0x15u )
        goto LABEL_15;
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v8 = sub_ADAB70(38, a2, a3, 0);
      else
        v8 = sub_AA93C0(0x26u, a2, (__int64)a3);
    }
    else
    {
      v8 = v7((__int64)v36, 38u, (_BYTE *)a2, (__int64)a3);
    }
    if ( v8 )
      goto LABEL_8;
LABEL_15:
    v28 = 257;
    v8 = sub_B51D30(38, a2, (__int64)a3, (__int64)v27, 0, 0);
    (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v37 + 2))(v37, v8, &v24, v33, v34);
    v18 = (unsigned __int64)v29;
    v23 = &v29[16 * (unsigned int)v30];
    if ( v29 != v23 )
    {
      do
      {
        v19 = *(_QWORD *)(v18 + 8);
        v20 = *(_DWORD *)v18;
        v18 += 16LL;
        sub_B99FD0(v8, v20, v19);
      }
      while ( v23 != (_BYTE *)v18 );
    }
    goto LABEL_8;
  }
  v8 = a2;
LABEL_8:
  v6[2] = v8;
  nullsub_61();
  v44 = &unk_49DA100;
  nullsub_63();
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
LABEL_10:
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(unsigned int *)(a1 + 12);
  v29 = v6;
  v11 = (char *)&v29;
  v12 = v6[2];
  v13 = *(_QWORD *)a1;
  v14 = v9 + 1;
  v15 = v9;
  if ( v9 + 1 > v10 )
  {
    if ( v13 > (unsigned __int64)&v29 || (unsigned __int64)&v29 >= v13 + 8 * v9 )
    {
      sub_2D57B00(a1, v14, v9, (__int64)&v29, v14, v5);
      v9 = *(unsigned int *)(a1 + 8);
      v13 = *(_QWORD *)a1;
      v11 = (char *)&v29;
      v15 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v21 = (char *)&v29 - v13;
      sub_2D57B00(a1, v14, v9, (__int64)&v29 - v13, v14, v5);
      v13 = *(_QWORD *)a1;
      v9 = *(unsigned int *)(a1 + 8);
      v11 = &v21[*(_QWORD *)a1];
      v15 = *(_DWORD *)(a1 + 8);
    }
  }
  v16 = (_QWORD *)(v13 + 8 * v9);
  if ( !v16 )
  {
    *(_DWORD *)(a1 + 8) = v15 + 1;
    goto LABEL_13;
  }
  *v16 = *(_QWORD *)v11;
  *(_QWORD *)v11 = 0;
  v6 = v29;
  ++*(_DWORD *)(a1 + 8);
  if ( v6 )
LABEL_13:
    (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
  return v12;
}
