// Function: sub_3910500
// Address: 0x3910500
//
__int64 __fastcall sub_3910500(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r14
  __int64 v5; // r12
  unsigned int v6; // ebx
  int *v7; // r13
  unsigned __int8 v8; // r14
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-90h]
  __int64 v12; // [rsp+10h] [rbp-80h]
  int *v13; // [rsp+18h] [rbp-78h]
  __int64 v14; // [rsp+20h] [rbp-70h]
  __int64 v15; // [rsp+28h] [rbp-68h]
  void (__fastcall *v16)(_QWORD *, __int64, unsigned __int64); // [rsp+30h] [rbp-60h]
  unsigned int v17; // [rsp+3Ch] [rbp-54h]
  _QWORD v18[2]; // [rsp+40h] [rbp-50h] BYREF
  char v19; // [rsp+50h] [rbp-40h]
  char v20; // [rsp+51h] [rbp-3Fh]

  result = *(unsigned int *)(a1 + 80);
  if ( (_DWORD)result )
  {
    v3 = a2[1];
    v20 = 1;
    v12 = v3;
    v18[0] = "filechecksums_begin";
    v19 = 3;
    v5 = sub_38BF8E0(v3, (__int64)v18, 0, 1);
    v20 = 1;
    v18[0] = "filechecksums_end";
    v19 = 3;
    v11 = sub_38BF8E0(v3, (__int64)v18, 0, 1);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 424LL))(a2, 244, 4);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a2 + 688LL))(a2, v11, v5, 4);
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v5, 0);
    v6 = 0;
    v7 = *(int **)(a1 + 72);
    v13 = &v7[8 * *(unsigned int *)(a1 + 80)];
    while ( v13 != v7 )
    {
      while ( 1 )
      {
        v8 = *((_BYTE *)v7 + 5);
        v9 = *((_QWORD *)v7 + 2);
        v17 = *v7;
        v14 = *((_QWORD *)v7 + 1);
        v15 = *((_QWORD *)v7 + 3);
        v16 = *(void (__fastcall **)(_QWORD *, __int64, unsigned __int64))(*a2 + 240LL);
        v10 = sub_38CB470(v6, v12);
        v16(a2, v15, v10);
        if ( v8 )
          break;
        v6 += 8;
        v7 += 8;
        (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, v17, 4);
        (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, 0, 4);
        if ( v13 == v7 )
          goto LABEL_7;
      }
      v7 += 8;
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, v17, 4);
      v6 = (v6 + v9 + 9) & 0xFFFFFFFC;
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, (unsigned __int8)v9, 1);
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 424LL))(a2, v8, 1);
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 400LL))(a2, v14, v9);
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, __int64, _QWORD))(*a2 + 512LL))(a2, 4, 0, 1, 0);
    }
LABEL_7:
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 176LL))(a2, v11, 0);
    *(_BYTE *)(a1 + 312) = 1;
    return a1;
  }
  return result;
}
