// Function: sub_921130
// Address: 0x921130
//
__int64 __fastcall sub_921130(
        unsigned int **a1,
        __int64 a2,
        __int64 a3,
        _BYTE **a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7)
{
  unsigned int *v11; // rdi
  __int64 (__fastcall *v12)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v14; // rax
  _BYTE **v15; // rcx
  __int64 v16; // r15
  __int64 v18; // r11
  __int64 v19; // rcx
  __int64 v20; // r12
  unsigned int *v21; // rbx
  unsigned int *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rsi
  _BYTE **v25; // rdi
  _BYTE **v26; // rax
  __int64 v27; // rsi
  int v28; // edx
  char v29; // al
  __int64 v30; // rax
  int v31; // [rsp+Ch] [rbp-84h]
  __int64 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-58h]
  __int64 v37; // [rsp+40h] [rbp-50h]
  unsigned int v38; // [rsp+48h] [rbp-48h]
  __int16 v39; // [rsp+50h] [rbp-40h]

  v11 = a1[10];
  v12 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v11 + 64LL);
  if ( v12 != sub_920540 )
  {
    v16 = v12((__int64)v11, a2, (_BYTE *)a3, a4, a5, a7);
    goto LABEL_6;
  }
  if ( !(unsigned __int8)sub_BCEA30(a2) && *(_BYTE *)a3 <= 0x15u )
  {
    v14 = sub_920370(a4, (__int64)&a4[a5]);
    if ( v15 == v14 )
    {
      LOBYTE(v39) = 0;
      v16 = sub_AD9FD0(a2, a3, (_DWORD)a4, a5, a7, (unsigned int)&v35, 0);
      if ( (_BYTE)v39 )
      {
        LOBYTE(v39) = 0;
        if ( v38 > 0x40 && v37 )
          j_j___libc_free_0_0(v37);
        if ( v36 > 0x40 && v35 )
          j_j___libc_free_0_0(v35);
      }
LABEL_6:
      if ( v16 )
        return v16;
    }
  }
  v39 = 257;
  v31 = a5 + 1;
  v16 = sub_BD2C40(88, (unsigned int)(a5 + 1));
  if ( v16 )
  {
    v18 = *(_QWORD *)(a3 + 8);
    v19 = v31 & 0x7FFFFFF;
    if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 > 1 )
    {
      v25 = &a4[a5];
      if ( v25 != a4 )
      {
        v26 = a4;
        while ( 1 )
        {
          v27 = *((_QWORD *)*v26 + 1);
          v28 = *(unsigned __int8 *)(v27 + 8);
          if ( v28 == 17 )
          {
            v29 = 0;
            goto LABEL_28;
          }
          if ( v28 == 18 )
            break;
          if ( v25 == ++v26 )
            goto LABEL_10;
        }
        v29 = 1;
LABEL_28:
        BYTE4(v34) = v29;
        LODWORD(v34) = *(_DWORD *)(v27 + 32);
        v30 = sub_BCE1B0(v18, v34);
        v19 = v31 & 0x7FFFFFF;
        v18 = v30;
      }
    }
LABEL_10:
    sub_B44260(v16, v18, 34, v19, 0, 0);
    *(_QWORD *)(v16 + 72) = a2;
    *(_QWORD *)(v16 + 80) = sub_B4DC50(a2, a4, a5);
    sub_B4D9A0(v16, a3, a4, a5, &v35);
  }
  sub_B4DDE0(v16, a7);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v16,
    a6,
    a1[7],
    a1[8]);
  v20 = 4LL * *((unsigned int *)a1 + 2);
  v21 = *a1;
  v22 = &v21[v20];
  while ( v22 != v21 )
  {
    v23 = *((_QWORD *)v21 + 1);
    v24 = *v21;
    v21 += 4;
    sub_B99FD0(v16, v24, v23);
  }
  return v16;
}
