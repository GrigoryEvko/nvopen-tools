// Function: sub_696CB0
// Address: 0x696cb0
//
__int64 __fastcall sub_696CB0(
        unsigned int a1,
        int a2,
        unsigned int a3,
        int a4,
        __int64 a5,
        _BYTE *a6,
        int a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 *a11,
        __int64 *a12,
        unsigned int *a13)
{
  unsigned __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r12d
  __int64 v27; // r15
  __int64 v28; // rax
  _BYTE *v29; // [rsp-10h] [rbp-100h]
  __int64 v30; // [rsp-8h] [rbp-F8h]
  int v32; // [rsp+14h] [rbp-DCh] BYREF
  __int64 v33; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v34[18]; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 v35; // [rsp+32h] [rbp-BEh]

  sub_6E1DD0(&v33);
  v18 = (unsigned __int64)v34;
  v19 = 2;
  sub_6E1E00(2, v34, 0, 0);
  v22 = qword_4F04C68;
  v23 = v35 | 1u;
  v35 |= 1u;
  v24 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(v24 + 12) & 0x10) != 0
    || !a10
    || ((v22 = &dword_4F04C44, dword_4F04C44 != -1) || (*(_BYTE *)(v24 + 6) & 2) != 0)
    && (v21 = dword_4F077BC) != 0
    && (v20 = (unsigned int)qword_4F077B4, !(_DWORD)qword_4F077B4)
    && qword_4F077A8 )
  {
    v23 = (unsigned int)v23 | 0xFFFFFF80;
    v35 = v23;
  }
  if ( a2 )
  {
    *a11 = sub_72CBE0(2, v34, v23, v22, v20, v21);
    *a13 = 0;
    if ( a8 )
    {
      v27 = sub_6E3060(a8);
      v25 = sub_84CF20(a5, a3, a4, a7, v27, a10, (__int64)a12, (__int64)a13);
      sub_6E1990(v27);
      v18 = (unsigned __int64)v29;
      v19 = v30;
    }
    else
    {
      v19 = a5;
      v18 = a3;
      v25 = sub_84CF20(a5, a3, a4, a7, a9, a10, (__int64)a12, (__int64)a13);
    }
    if ( v25 )
      *a11 = *a12;
  }
  else if ( a1 )
  {
    if ( a9 )
    {
      if ( *(_BYTE *)(a9 + 8) == 1 )
      {
        v25 = 0;
        *a13 = 0;
        goto LABEL_19;
      }
      a8 = *(_QWORD *)(a9 + 24) + 8LL;
    }
    if ( (*(_BYTE *)(a8 + 19) & 8) != 0 && *(_BYTE *)(a8 + 16) == 3 )
      sub_6F6890(a8, 0);
    v18 = (unsigned __int64)&v32;
    v19 = sub_6968F0(a8, &v32);
    *a12 = v19;
    if ( a7 )
    {
      v19 = sub_7259C0(12);
      v28 = *a12;
      *(_BYTE *)(v19 + 184) = 2;
      *(_QWORD *)(v19 + 160) = v28;
    }
    *a11 = v19;
    v25 = 0;
    if ( (unsigned int)sub_8DBE70(v19) )
    {
      *a13 = a1;
    }
    else
    {
      v25 = a1;
      *a13 = 0;
    }
  }
  else
  {
    v19 = a5;
    v18 = (unsigned __int64)a6;
    v25 = sub_83D110(a5, (_DWORD)a6, a7, a8, a9, a10, (__int64)a11, (__int64)a12, (__int64)a13);
  }
LABEL_19:
  sub_6E2B30(v19, v18);
  sub_6E1DF0(v33);
  return v25;
}
