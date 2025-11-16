// Function: sub_E5FB60
// Address: 0xe5fb60
//
__int64 __fastcall sub_E5FB60(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r14
  __int64 v5; // r12
  unsigned int v6; // ebx
  int *v7; // r13
  unsigned __int8 v8; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-A0h]
  __int64 v12; // [rsp+10h] [rbp-90h]
  int *v13; // [rsp+18h] [rbp-88h]
  __int64 v14; // [rsp+20h] [rbp-80h]
  __int64 v15; // [rsp+28h] [rbp-78h]
  void (__fastcall *v16)(_QWORD *, __int64, __int64); // [rsp+30h] [rbp-70h]
  unsigned int v17; // [rsp+3Ch] [rbp-64h]
  _QWORD v18[4]; // [rsp+40h] [rbp-60h] BYREF
  char v19; // [rsp+60h] [rbp-40h]
  char v20; // [rsp+61h] [rbp-3Fh]

  result = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)result )
  {
    v3 = a2[1];
    v20 = 1;
    v12 = v3;
    v18[0] = "filechecksums_begin";
    v19 = 3;
    v5 = sub_E6C380(v3, v18, 0);
    v20 = 1;
    v18[0] = "filechecksums_end";
    v19 = 3;
    v11 = sub_E6C380(v3, v18, 0);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, 244, 4);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 832LL))(a2, v11, v5, 4);
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v5, 0);
    v6 = 0;
    v7 = *(int **)(a1 + 64);
    v13 = &v7[8 * *(unsigned int *)(a1 + 72)];
    while ( v13 != v7 )
    {
      while ( 1 )
      {
        v8 = *((_BYTE *)v7 + 5);
        v9 = *((_QWORD *)v7 + 2);
        v17 = *v7;
        v14 = *((_QWORD *)v7 + 1);
        v15 = *((_QWORD *)v7 + 3);
        v16 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 272LL);
        v10 = sub_E81A90(v6, v12, 0, 0);
        v16(a2, v15, v10);
        if ( v8 )
          break;
        v6 += 8;
        v7 += 8;
        (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, v17, 4);
        (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, 0, 4);
        if ( v13 == v7 )
          goto LABEL_7;
      }
      v7 += 8;
      v6 = 4 * ((v6 + (_DWORD)v9 != -6) + ((v6 + (unsigned int)v9 + 6 - (v6 + (_DWORD)v9 != -6)) >> 2));
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, v17, 4);
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, (unsigned __int8)v9, 1);
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, v8, 1);
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 512LL))(a2, v14, v9);
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, _QWORD))(*a2 + 608LL))(a2, 2, 0, 1, 0);
    }
LABEL_7:
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v11, 0);
    *(_BYTE *)(a1 + 304) = 1;
    return a1;
  }
  return result;
}
