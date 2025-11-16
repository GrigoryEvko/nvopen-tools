// Function: sub_F5C6D0
// Address: 0xf5c6d0
//
__int64 __fastcall sub_F5C6D0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // r15
  int v7; // ebx
  _QWORD *v8; // rax
  unsigned __int8 *v9; // rdi
  void (__fastcall *v10)(_BYTE *, __int64, __int64); // rax
  unsigned int v13; // [rsp+14h] [rbp-5Ch]
  _QWORD *v14; // [rsp+18h] [rbp-58h]
  _BYTE v15[16]; // [rsp+20h] [rbp-50h] BYREF
  void (__fastcall *v16)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-40h]
  __int64 v17; // [rsp+38h] [rbp-38h]

  v13 = *(_DWORD *)(a1 + 8);
  if ( !v13 )
    return 0;
  v6 = 0;
  v7 = 0;
  do
  {
    v8 = (_QWORD *)(v6 + *(_QWORD *)a1);
    v9 = (unsigned __int8 *)v8[2];
    if ( v9 )
    {
      if ( *v9 <= 0x1Cu )
        goto LABEL_8;
      if ( sub_F50EE0(v9, 0) )
        goto LABEL_12;
      v8 = (_QWORD *)(v6 + *(_QWORD *)a1);
      v9 = (unsigned __int8 *)v8[2];
      if ( v9 )
      {
LABEL_8:
        if ( v9 != (unsigned __int8 *)-8192LL && v9 != (unsigned __int8 *)-4096LL )
        {
          v14 = v8;
          sub_BD60C0(v8);
          v8 = v14;
        }
        v8[2] = 0;
      }
    }
    ++v7;
LABEL_12:
    v6 += 24;
  }
  while ( 24LL * v13 != v6 );
  if ( v7 == v13 )
    return 0;
  v10 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 16);
  v16 = 0;
  if ( v10 )
  {
    v10(v15, a4, 2);
    v17 = *(_QWORD *)(a4 + 24);
    v16 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a4 + 16);
  }
  sub_F5C330(a1, a2, a3, (__int64)v15);
  if ( v16 )
    v16(v15, v15, 3);
  return 1;
}
