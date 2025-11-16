// Function: sub_182BFF0
// Address: 0x182bff0
//
__int64 __fastcall sub_182BFF0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r13
  __int64 *v5; // rdi
  size_t v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 *v9; // rdi
  int v10; // eax
  __int64 v11; // rax
  char v12; // r15
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  _DWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  _DWORD *v19; // r8
  _DWORD *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rax
  int v27; // r9d
  bool v28; // zf
  __int64 v30; // rax
  size_t v31; // rdx
  __int64 *v32; // [rsp+0h] [rbp-A0h] BYREF
  __int16 v33; // [rsp+10h] [rbp-90h]
  __int64 *p_src; // [rsp+20h] [rbp-80h] BYREF
  size_t n; // [rsp+28h] [rbp-78h] BYREF
  __int64 src; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v37; // [rsp+38h] [rbp-68h]
  __int64 v38; // [rsp+40h] [rbp-60h]
  __int64 v39; // [rsp+48h] [rbp-58h]
  __int64 v40; // [rsp+50h] [rbp-50h]
  __int64 v41; // [rsp+58h] [rbp-48h]

  v4 = sub_1632FA0((__int64)a2);
  v33 = 260;
  v32 = a2 + 30;
  sub_16E1010((__int64)&p_src, (__int64)&v32);
  v5 = *(__int64 **)(a1 + 168);
  if ( p_src == &src )
  {
    v31 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v5 = src;
      else
        memcpy(v5, &src, n);
      v31 = n;
      v5 = *(__int64 **)(a1 + 168);
    }
    *(_QWORD *)(a1 + 176) = v31;
    *((_BYTE *)v5 + v31) = 0;
    v5 = p_src;
  }
  else
  {
    v6 = n;
    v7 = src;
    if ( v5 == (__int64 *)(a1 + 184) )
    {
      *(_QWORD *)(a1 + 168) = p_src;
      *(_QWORD *)(a1 + 176) = v6;
      *(_QWORD *)(a1 + 184) = v7;
    }
    else
    {
      v8 = *(_QWORD *)(a1 + 184);
      *(_QWORD *)(a1 + 168) = p_src;
      *(_QWORD *)(a1 + 176) = v6;
      *(_QWORD *)(a1 + 184) = v7;
      if ( v5 )
      {
        p_src = v5;
        src = v8;
        goto LABEL_5;
      }
    }
    p_src = &src;
    v5 = &src;
  }
LABEL_5:
  n = 0;
  *(_BYTE *)v5 = 0;
  v9 = p_src;
  v10 = v40;
  *(_QWORD *)(a1 + 200) = v38;
  *(_QWORD *)(a1 + 208) = v39;
  *(_QWORD *)(a1 + 216) = v40;
  if ( v9 != &src )
  {
    j_j___libc_free_0(v9, src + 1);
    v10 = *(_DWORD *)(a1 + 216);
  }
  if ( v10 == 10
    && ((sub_16E22F0(a1 + 168, &p_src, (_DWORD *)&p_src + 1, &n), sub_16E2900(a1 + 168)) || (unsigned int)p_src > 0x14) )
  {
    v12 = byte_4FA9B60;
    v11 = 0;
    *(_DWORD *)(a1 + 224) = 4;
    if ( !v12 )
    {
      v12 = 1;
      v11 = -(__int64)((unsigned __int8)byte_4FAA340 ^ 1u);
    }
  }
  else
  {
    *(_DWORD *)(a1 + 224) = 4;
    v11 = 0;
    v12 = 0;
  }
  *(_QWORD *)(a1 + 232) = v11;
  v13 = sub_16D5D50();
  v14 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v15 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v16 = v14[2];
        v17 = v14[3];
        if ( v13 <= v14[4] )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v17 )
          goto LABEL_14;
      }
      v15 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v16 );
LABEL_14:
    if ( v15 != dword_4FA0208 && v13 >= *((_QWORD *)v15 + 4) )
    {
      v18 = *((_QWORD *)v15 + 7);
      v19 = v15 + 12;
      if ( v18 )
      {
        v20 = v15 + 12;
        do
        {
          while ( 1 )
          {
            v21 = *(_QWORD *)(v18 + 16);
            v22 = *(_QWORD *)(v18 + 24);
            if ( *(_DWORD *)(v18 + 32) >= dword_4FA99E8 )
              break;
            v18 = *(_QWORD *)(v18 + 24);
            if ( !v22 )
              goto LABEL_21;
          }
          v20 = (_DWORD *)v18;
          v18 = *(_QWORD *)(v18 + 16);
        }
        while ( v21 );
LABEL_21:
        if ( v20 != v19 && dword_4FA99E8 >= v20[8] && (int)v20[9] > 0 )
          *(_QWORD *)(a1 + 232) = qword_4FA9A80;
      }
    }
  }
  *(_BYTE *)(a1 + 240) = v12;
  v23 = *a2;
  p_src = 0;
  *(_QWORD *)(a1 + 160) = v23;
  v37 = (_QWORD *)v23;
  src = 0;
  v38 = 0;
  LODWORD(v39) = 0;
  v40 = 0;
  v41 = 0;
  n = 0;
  v24 = sub_15A9620(v4, v23, 0);
  v25 = v37;
  *(_QWORD *)(a1 + 248) = v24;
  v26 = sub_1643330(v25);
  v28 = *(_BYTE *)(a1 + 264) == 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 256) = v26;
  if ( v28 )
  {
    v30 = sub_1B281E0(
            (_DWORD)a2,
            (unsigned int)"hwasan.module_ctor",
            18,
            (unsigned int)"__hwasan_init",
            13,
            v27,
            0,
            0,
            0,
            0,
            0,
            0);
    *(_QWORD *)(a1 + 272) = v30;
    sub_1B28000(a2, v30, 0, 0);
  }
  if ( p_src )
    sub_161E7C0((__int64)&p_src, (__int64)p_src);
  return 1;
}
