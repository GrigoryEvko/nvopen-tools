// Function: sub_221F630
// Address: 0x221f630
//
char *__fastcall sub_221F630(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        _DWORD *a7,
        _DWORD *a8,
        char a9,
        char a10)
{
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdi
  char v17; // al
  char *v18; // r13
  int v19; // edx
  char v20; // bl
  char v21; // r14
  char v22; // al
  __int64 (__fastcall *v24)(__int64, unsigned int); // rdx
  char s; // [rsp+3Ch] [rbp-3Ch] BYREF
  char v27; // [rsp+3Dh] [rbp-3Bh]
  char v28; // [rsp+3Eh] [rbp-3Ah]
  char v29; // [rsp+3Fh] [rbp-39h]

  v14 = sub_222F790(a6 + 208);
  v15 = a6;
  v16 = v14;
  *a7 = 0;
  if ( *(_BYTE *)(v14 + 56) )
  {
    v17 = *(_BYTE *)(v14 + 94);
  }
  else
  {
    sub_2216D60(v14);
    v15 = a6;
    v24 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v16 + 48LL);
    v17 = 37;
    if ( v24 != sub_CE72A0 )
    {
      v17 = v24(v16, 37u);
      v15 = a6;
    }
  }
  s = v17;
  if ( a10 )
  {
    v27 = a10;
    v29 = 0;
    v28 = a9;
  }
  else
  {
    v28 = 0;
    v27 = a9;
  }
  v18 = sub_221D830(a1, a2, a3, (char *)a4, a5, v15, a7, a8, &s);
  v20 = v19 == -1;
  v21 = v20 & (v18 != 0);
  if ( v21 )
  {
    v20 = 0;
    if ( *((_QWORD *)v18 + 2) >= *((_QWORD *)v18 + 3)
      && (*(unsigned int (__fastcall **)(char *))(*(_QWORD *)v18 + 72LL))(v18) == -1 )
    {
      v20 = v21;
      v18 = 0;
    }
  }
  v22 = a5 == -1;
  if ( a4 )
  {
    if ( a5 == -1 )
    {
      v22 = 0;
      if ( *(_QWORD *)(a4 + 16) >= *(_QWORD *)(a4 + 24) )
        v22 = (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)a4 + 72LL))(a4) == -1;
    }
  }
  if ( v20 == v22 )
    *a7 |= 2u;
  return v18;
}
