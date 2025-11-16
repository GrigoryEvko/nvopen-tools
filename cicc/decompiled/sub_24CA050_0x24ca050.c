// Function: sub_24CA050
// Address: 0x24ca050
//
__int64 __fastcall sub_24CA050(__int64 a1, _QWORD *a2, char *a3, const char *a4, _QWORD *a5, const char *a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // r15d
  int v11; // eax
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  size_t v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v19; // rdx
  char v20; // al
  _QWORD v21[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v22[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h] BYREF

  v22[0] = sub_24C9AA0(a1, a2, a6, a5);
  v8 = *(_QWORD *)(a1 + 456);
  v22[1] = v9;
  v21[0] = v8;
  v21[1] = v8;
  v10 = strlen(a4);
  v11 = strlen(a3);
  sub_2A41510(
    (unsigned int)&v23,
    (_DWORD)a2,
    (_DWORD)a3,
    v11,
    (_DWORD)a4,
    v10,
    (__int64)v21,
    2,
    (__int64)v22,
    2,
    0,
    0,
    0);
  v12 = *(unsigned int *)(a1 + 604);
  v13 = v23;
  if ( (unsigned int)v12 <= 8 && (v19 = 292, _bittest64(&v19, v12)) )
  {
    sub_2A3ED40(a2, v23, 2, 0);
    if ( *(_DWORD *)(a1 + 604) != 1 )
      return v13;
  }
  else
  {
    v14 = strlen(a3);
    v15 = sub_BAA410((__int64)a2, a3, v14);
    sub_B2F990(v13, v15, v16, v17);
    sub_2A3ED40(a2, v13, 2, v13);
    if ( *(_DWORD *)(a1 + 604) != 1 )
      return v13;
  }
  v20 = *(_BYTE *)(v13 + 32) & 0xF0 | 5;
  *(_BYTE *)(v13 + 32) = v20;
  if ( (v20 & 0x30) != 0 )
    *(_BYTE *)(v13 + 33) |= 0x40u;
  return v13;
}
