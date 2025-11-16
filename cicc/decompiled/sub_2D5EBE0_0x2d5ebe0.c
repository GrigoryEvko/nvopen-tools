// Function: sub_2D5EBE0
// Address: 0x2d5ebe0
//
__int64 __fastcall sub_2D5EBE0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 **a4)
{
  _QWORD *v6; // rax
  __int64 v7; // r9
  _QWORD *v8; // r12
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  char *v13; // rcx
  __int64 v14; // r15
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // r8
  int v17; // edx
  _QWORD *v18; // rax
  _QWORD *v20; // rax
  unsigned __int64 v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // esi
  char *v24; // r13
  _BYTE *v26; // [rsp+18h] [rbp-128h]
  char *v27; // [rsp+20h] [rbp-120h] BYREF
  char v28; // [rsp+40h] [rbp-100h]
  char v29; // [rsp+41h] [rbp-FFh]
  _QWORD v30[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v31; // [rsp+70h] [rbp-D0h]
  _BYTE *v32; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+88h] [rbp-B8h]
  _BYTE v34[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+B0h] [rbp-90h]
  __int64 v36; // [rsp+B8h] [rbp-88h]
  __int64 v37; // [rsp+C0h] [rbp-80h]
  __int64 v38; // [rsp+C8h] [rbp-78h]
  void **v39; // [rsp+D0h] [rbp-70h]
  void **v40; // [rsp+D8h] [rbp-68h]
  __int64 v41; // [rsp+E0h] [rbp-60h]
  int v42; // [rsp+E8h] [rbp-58h]
  __int16 v43; // [rsp+ECh] [rbp-54h]
  char v44; // [rsp+EEh] [rbp-52h]
  __int64 v45; // [rsp+F0h] [rbp-50h]
  __int64 v46; // [rsp+F8h] [rbp-48h]
  void *v47; // [rsp+100h] [rbp-40h] BYREF
  void *v48; // [rsp+108h] [rbp-38h] BYREF

  v6 = (_QWORD *)sub_22077B0(0x18u);
  v8 = v6;
  if ( !v6 )
    goto LABEL_10;
  v6[1] = a2;
  *v6 = off_49D40F0;
  v38 = sub_BD5C60(a2);
  v39 = &v47;
  v40 = &v48;
  v32 = v34;
  v47 = &unk_49DA100;
  v33 = 0x200000000LL;
  v43 = 512;
  LOWORD(v37) = 0;
  v48 = &unk_49DA0B0;
  v41 = 0;
  v42 = 0;
  v44 = 7;
  v45 = 0;
  v46 = 0;
  v35 = 0;
  v36 = 0;
  sub_D5F1F0((__int64)&v32, a2);
  v30[0] = 0;
  sub_93FB40((__int64)&v32, 0);
  v29 = 1;
  v27 = "promoted";
  v28 = 3;
  if ( a4 != *(__int64 ***)(a3 + 8) )
  {
    v9 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v39 + 15);
    if ( v9 == sub_920130 )
    {
      if ( *(_BYTE *)a3 > 0x15u )
        goto LABEL_15;
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v10 = sub_ADAB70(39, a3, a4, 0);
      else
        v10 = sub_AA93C0(0x27u, a3, (__int64)a4);
    }
    else
    {
      v10 = v9((__int64)v39, 39u, (_BYTE *)a3, (__int64)a4);
    }
    if ( v10 )
      goto LABEL_8;
LABEL_15:
    v31 = 257;
    v20 = sub_BD2C40(72, 1u);
    v10 = (__int64)v20;
    if ( v20 )
      sub_B515B0((__int64)v20, a3, (__int64)a4, (__int64)v30, 0, 0);
    (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v40 + 2))(v40, v10, &v27, v36, v37);
    v21 = (unsigned __int64)v32;
    v26 = &v32[16 * (unsigned int)v33];
    if ( v32 != v26 )
    {
      do
      {
        v22 = *(_QWORD *)(v21 + 8);
        v23 = *(_DWORD *)v21;
        v21 += 16LL;
        sub_B99FD0(v10, v23, v22);
      }
      while ( v26 != (_BYTE *)v21 );
    }
    goto LABEL_8;
  }
  v10 = a3;
LABEL_8:
  v8[2] = v10;
  nullsub_61();
  v47 = &unk_49DA100;
  nullsub_63();
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
LABEL_10:
  v11 = *(unsigned int *)(a1 + 8);
  v12 = *(unsigned int *)(a1 + 12);
  v32 = v8;
  v13 = (char *)&v32;
  v14 = v8[2];
  v15 = *(_QWORD *)a1;
  v16 = v11 + 1;
  v17 = v11;
  if ( v11 + 1 > v12 )
  {
    if ( v15 > (unsigned __int64)&v32 || (unsigned __int64)&v32 >= v15 + 8 * v11 )
    {
      sub_2D57B00(a1, v16, v11, (__int64)&v32, v16, v7);
      v11 = *(unsigned int *)(a1 + 8);
      v15 = *(_QWORD *)a1;
      v13 = (char *)&v32;
      v17 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v24 = (char *)&v32 - v15;
      sub_2D57B00(a1, v16, v11, (__int64)&v32 - v15, v16, v7);
      v15 = *(_QWORD *)a1;
      v11 = *(unsigned int *)(a1 + 8);
      v13 = &v24[*(_QWORD *)a1];
      v17 = *(_DWORD *)(a1 + 8);
    }
  }
  v18 = (_QWORD *)(v15 + 8 * v11);
  if ( !v18 )
  {
    *(_DWORD *)(a1 + 8) = v17 + 1;
    goto LABEL_13;
  }
  *v18 = *(_QWORD *)v13;
  *(_QWORD *)v13 = 0;
  v8 = v32;
  ++*(_DWORD *)(a1 + 8);
  if ( v8 )
LABEL_13:
    (*(void (__fastcall **)(_QWORD *))(*v8 + 8LL))(v8);
  return v14;
}
