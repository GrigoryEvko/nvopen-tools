// Function: sub_E29610
// Address: 0xe29610
//
void *__fastcall sub_E29610(size_t a1, size_t a2, _QWORD *a3, _DWORD *a4, char a5)
{
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rax
  char v11; // cl
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rsi
  void *v15; // rax
  void *v16; // r13
  _QWORD *v17; // rbx
  _QWORD *v19; // [rsp+8h] [rbp-148h]
  size_t v20[2]; // [rsp+10h] [rbp-140h] BYREF
  void *src; // [rsp+20h] [rbp-130h] BYREF
  __int64 v22; // [rsp+28h] [rbp-128h]
  unsigned __int64 v23; // [rsp+30h] [rbp-120h]
  __int64 v24; // [rsp+38h] [rbp-118h]
  int v25; // [rsp+40h] [rbp-110h]
  void *i; // [rsp+50h] [rbp-100h] BYREF
  char v27; // [rsp+58h] [rbp-F8h]
  _QWORD *v28; // [rsp+60h] [rbp-F0h]
  __int64 v29; // [rsp+B8h] [rbp-98h]
  __int64 v30; // [rsp+110h] [rbp-40h]

  v27 = 0;
  v28 = 0;
  i = &unk_49E0E68;
  v8 = (_QWORD *)sub_22077B0(32);
  if ( v8 )
  {
    *v8 = 0;
    v8[1] = 0;
    v8[2] = 0;
    v8[3] = 0;
  }
  v19 = v8;
  v20[1] = a2;
  *v8 = sub_2207820(4096);
  v9 = v28;
  v19[2] = 4096;
  v19[3] = v9;
  v19[1] = 0;
  v28 = v19;
  v29 = 0;
  v30 = 0;
  v20[0] = a1;
  v10 = sub_E25DD0((__int64)&i, v20);
  v11 = v27;
  v12 = v10;
  if ( a3 && v27 != 1 )
    *a3 = a1 - v20[0];
  if ( (a5 & 1) != 0 )
  {
    sub_E243B0((__int64)&i);
    v11 = v27;
  }
  v13 = ((a5 & 4) != 0) | 4u;
  if ( (a5 & 2) == 0 )
    v13 = (a5 & 4) != 0;
  if ( (a5 & 8) != 0 )
    v13 = (unsigned int)v13 | 0x10;
  if ( (a5 & 0x10) != 0 )
    v13 = (unsigned int)v13 | 8;
  if ( (a5 & 0x20) != 0 )
    v13 = (unsigned int)v13 | 0x20;
  if ( v11 )
  {
    if ( a4 )
      *a4 = -2;
    v16 = 0;
  }
  else
  {
    src = 0;
    v22 = 0;
    v23 = 0;
    v24 = -1;
    v25 = 1;
    (*(void (__fastcall **)(__int64, void **, __int64))(*(_QWORD *)v12 + 16LL))(v12, &src, v13);
    v14 = v22;
    if ( v22 + 1 > v23 )
    {
      if ( v22 + 993 > 2 * v23 )
        v23 = v22 + 993;
      else
        v23 *= 2LL;
      v15 = (void *)realloc(src);
      src = v15;
      if ( !v15 )
        abort();
      v14 = v22;
    }
    else
    {
      v15 = src;
    }
    *((_BYTE *)v15 + v14) = 0;
    v16 = src;
    if ( a4 )
      *a4 = 0;
  }
  v17 = v28;
  for ( i = &unk_49E0E68; v17; v28 = v17 )
  {
    if ( *v17 )
      j_j___libc_free_0_0(*v17);
    v17 = (_QWORD *)v28[3];
    j_j___libc_free_0(v28, 32);
  }
  return v16;
}
