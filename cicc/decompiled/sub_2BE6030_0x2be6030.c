// Function: sub_2BE6030
// Address: 0x2be6030
//
__int64 __fastcall sub_2BE6030(__int64 a1, _QWORD *a2, unsigned __int8 *a3, unsigned __int8 *a4)
{
  _QWORD *v5; // r14
  _BYTE *v7; // r12
  _QWORD *v8; // r11
  unsigned __int8 *v9; // rbx
  unsigned __int8 *v10; // r14
  __int64 (__fastcall *v11)(__int64, unsigned int); // rcx
  unsigned __int64 v12; // rdx
  size_t v13; // r15
  unsigned __int64 v14; // rcx
  size_t v15; // rax
  __int64 v16; // r15
  unsigned __int8 v17; // al
  unsigned __int8 v18; // r13
  char **v19; // r15
  __int64 v20; // rdx
  unsigned int v21; // r13d
  __int64 (__fastcall *v23)(__int64, unsigned int); // rax
  _QWORD *v24; // [rsp+10h] [rbp-60h]
  size_t v25; // [rsp+18h] [rbp-58h]
  _QWORD *v26; // [rsp+18h] [rbp-58h]
  _QWORD *v27; // [rsp+20h] [rbp-50h] BYREF
  size_t v28; // [rsp+28h] [rbp-48h]
  _QWORD v29[8]; // [rsp+30h] [rbp-40h] BYREF

  v5 = v29;
  v27 = v29;
  v28 = 0;
  v7 = (_BYTE *)sub_222F790(a2, (__int64)a2);
  LOBYTE(v29[0]) = 0;
  if ( a3 != a4 )
  {
    v8 = v29;
    v9 = a3;
    v10 = a4;
    do
    {
      v16 = *v9;
      v18 = *v9;
      if ( v7[v16 + 313] )
      {
        v18 = v7[v16 + 313];
      }
      else
      {
        v11 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v7 + 64LL);
        if ( v11 != sub_2216C50 )
        {
          v26 = v8;
          v17 = ((__int64 (__fastcall *)(_BYTE *, _QWORD, _QWORD))v11)(v7, (unsigned int)(char)v16, 0);
          v8 = v26;
          v18 = v17;
        }
        if ( v18 )
          v7[v16 + 313] = v18;
      }
      v12 = (unsigned __int64)v27;
      v13 = v28;
      v14 = 15;
      if ( v27 != v8 )
        v14 = v29[0];
      v15 = v28 + 1;
      if ( v28 + 1 > v14 )
      {
        v24 = v8;
        v25 = v28 + 1;
        sub_2240BB0((unsigned __int64 *)&v27, v28, 0, 0, 1u);
        v12 = (unsigned __int64)v27;
        v8 = v24;
        v15 = v25;
      }
      *(_BYTE *)(v12 + v13) = v18;
      ++v9;
      v28 = v15;
      *((_BYTE *)v27 + v13 + 1) = 0;
    }
    while ( v10 != v9 );
    v5 = v8;
  }
  v19 = (char **)&off_4CDFC60;
  while ( sub_2241AC0((__int64)&v27, *v19) )
  {
    if ( &s == ++v19 )
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_BYTE *)(a1 + 16) = 0;
      goto LABEL_21;
    }
  }
  v20 = ((char *)v19 - (char *)&off_4CDFC60) >> 3;
  if ( v7[56] )
  {
    LOBYTE(v21) = v7[(unsigned __int8)v20 + 57];
  }
  else
  {
    v21 = (char)v20;
    sub_2216D60((__int64)v7);
    v23 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v7 + 48LL);
    if ( v23 != sub_CE72A0 )
      LOBYTE(v21) = v23((__int64)v7, v21);
  }
  *(_QWORD *)a1 = a1 + 16;
  sub_2240A50((__int64 *)a1, 1u, v21);
LABEL_21:
  if ( v27 != v5 )
    j_j___libc_free_0((unsigned __int64)v27);
  return a1;
}
