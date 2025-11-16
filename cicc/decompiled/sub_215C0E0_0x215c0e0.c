// Function: sub_215C0E0
// Address: 0x215c0e0
//
void __fastcall sub_215C0E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 v5; // r15
  char v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 (*v12)(); // rax
  __int64 v13; // rbx
  __int64 v14; // rdx
  _BYTE *v15; // r11
  _BYTE *v16; // r9
  size_t v17; // r10
  unsigned __int64 v18; // r15
  _QWORD *v19; // rdi
  int v20; // eax
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // r15
  int v24; // r8d
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int i; // r12d
  __int64 v28; // r8
  size_t v29; // [rsp+10h] [rbp-130h]
  _BYTE *v30; // [rsp+18h] [rbp-128h]
  _BYTE *v31; // [rsp+20h] [rbp-120h]
  __int64 j; // [rsp+38h] [rbp-108h]
  _QWORD v36[2]; // [rsp+40h] [rbp-100h] BYREF
  _QWORD *v37; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int64 v38; // [rsp+58h] [rbp-E8h]
  _QWORD dest[8]; // [rsp+60h] [rbp-E0h] BYREF
  char v40[8]; // [rsp+A0h] [rbp-A0h] BYREF
  char *v41; // [rsp+A8h] [rbp-98h]
  char v42; // [rsp+B8h] [rbp-88h] BYREF
  int v43; // [rsp+ECh] [rbp-54h]
  __int64 v44; // [rsp+108h] [rbp-38h]

  v5 = HIDWORD(a5);
  if ( HIDWORD(a5) )
  {
    v6 = a5;
    v7 = sub_145CDC0(0x10u, a1 + 52);
    v8 = v7;
    if ( v7 )
    {
      *(_QWORD *)v7 = 0;
      *(_DWORD *)(v7 + 8) = 0;
    }
    v9 = a1[1];
    sub_39A1E10(v40, v9, a2, v7);
    if ( !v6 )
      v43 = 2;
    if ( (v5 & 0x80000000) != 0LL )
    {
      v25 = *a1;
      v37 = dest;
      v38 = 0;
      LOBYTE(dest[0]) = 0;
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD **))(v25 + 160))(a1, (unsigned int)v5, &v37);
      LODWORD(v36[0]) = 65547;
      sub_39A3560(a2, v8, 0, v36, 146);
      v26 = 0;
      if ( v38 )
      {
        for ( i = 0; i < v38; v26 = ++i )
        {
          v28 = *((unsigned __int8 *)v37 + v26);
          LODWORD(v36[0]) = 65547;
          sub_39A3560(a2, v8, 0, v36, v28);
        }
      }
      LODWORD(v36[0]) = 65551;
      sub_39A3560(a2, v8, 0, v36, 0);
      BYTE2(v36[0]) = 0;
      sub_39A3560(a2, a4 + 8, 51, v36, 12);
      if ( v37 != dest )
        j_j___libc_free_0(v37, dest[0] + 1LL);
    }
    else
    {
      v10 = a1[1];
      v11 = 0;
      v12 = *(__int64 (**)())(*(_QWORD *)v10 + 184LL);
      if ( v12 != sub_215B780 )
        v11 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v12)(v10, v9, 0);
      sub_39A39D0(a2, v8, v11);
      BYTE2(v37) = 0;
      sub_39A3560(a2, a4 + 8, 51, &v37, 6);
    }
    v13 = sub_39888D0(a3);
    for ( j = v13 + 16 * v14; v13 != j; v13 += 16 )
    {
      v23 = *(_QWORD *)(v13 + 8);
      sub_399FD50(v40, v23);
      v37 = dest;
      v38 = 0x800000000LL;
      if ( v23 )
      {
        v15 = *(_BYTE **)(v23 + 32);
        v16 = *(_BYTE **)(v23 + 24);
        v17 = v15 - v16;
        v18 = (v15 - v16) >> 3;
        if ( (unsigned __int64)(v15 - v16) > 0x40 )
        {
          v29 = v15 - v16;
          v30 = v15;
          v31 = v16;
          sub_16CD150((__int64)&v37, dest, v18, 8, v24, (int)v16);
          v21 = v37;
          v20 = v38;
          v16 = v31;
          v15 = v30;
          v17 = v29;
          v19 = &v37[(unsigned int)v38];
        }
        else
        {
          v19 = dest;
          v20 = 0;
          v21 = dest;
        }
        if ( v16 != v15 )
        {
          memcpy(v19, v16, v17);
          v20 = v38;
          v21 = v37;
        }
        LODWORD(v38) = v18 + v20;
        v22 = &v21[(unsigned int)(v18 + v20)];
      }
      else
      {
        v21 = dest;
        v22 = dest;
      }
      v36[0] = v21;
      v36[1] = v22;
      sub_399FAC0(v40, v36, 0);
      if ( v37 != dest )
        _libc_free((unsigned __int64)v37);
    }
    sub_399FD30(v40);
    sub_39A4520(a2, a4, 2, v44);
    if ( v41 != &v42 )
      _libc_free((unsigned __int64)v41);
  }
}
