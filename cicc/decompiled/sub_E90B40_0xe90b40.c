// Function: sub_E90B40
// Address: 0xe90b40
//
__int64 __fastcall sub_E90B40(unsigned int *a1, _QWORD *a2, __int64 a3)
{
  char v6; // r14
  char v7; // si
  char v8; // r14
  __int64 v9; // rax
  unsigned __int8 v10; // si
  _QWORD *v11; // r14
  __int64 v12; // r15
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 result; // rax
  __int64 v22; // rsi
  _QWORD v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *((_BYTE *)a1 + 8);
  sub_E98EB0(a2, *a1, 0);
  v7 = *((_BYTE *)a1 + 8);
  v8 = v6 & 2;
  if ( a1[1] )
    v7 |= 4u;
  v9 = *a2;
  v10 = *((_BYTE *)a1 + 9) | (16 * v7);
  if ( v8 )
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(v9 + 536))(a2, v10, 1);
    result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, *((_QWORD *)a1 + 2), 8);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(v9 + 536))(a2, v10 | 0x80u, 1);
    v11 = (_QWORD *)a2[1];
    v12 = *(_QWORD *)(a3 + 24);
    v13 = sub_E808D0(*((_QWORD *)a1 + 3), 0, v11, 0);
    v14 = sub_E808D0(v12, 0, v11, 0);
    v15 = sub_E81A00(18, v13, v14, v11, 0);
    v16 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
    if ( !sub_E81930(v15, v23, v16) )
    {
      v17 = (_QWORD *)a2[1];
      v18 = v17[36];
      v17[46] += 120LL;
      v19 = (v18 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v17[37] >= (unsigned __int64)(v19 + 120) && v18 )
        v17[36] = v19 + 120;
      else
        v19 = sub_9D1E70((__int64)(v17 + 36), 120, 120, 3);
      sub_E81B30(v19, 13, 0);
      *(_BYTE *)(v19 + 30) = 0;
      *(_QWORD *)(v19 + 40) = v19 + 64;
      *(_QWORD *)(v19 + 72) = v19 + 88;
      *(_QWORD *)(v19 + 32) = 0;
      *(_QWORD *)(v19 + 48) = 0;
      *(_QWORD *)(v19 + 56) = 8;
      *(_QWORD *)(v19 + 80) = 0x100000000LL;
      *(_QWORD *)(v19 + 112) = v15;
      v20 = *(_QWORD *)(a2[36] + 8LL);
      *(_QWORD *)(v19 + 8) = v20;
      *(_DWORD *)(v19 + 24) = *(_DWORD *)(a2[36] + 24LL) + 1;
      *(_QWORD *)a2[36] = v19;
      a2[36] = v19;
      result = *(_QWORD *)(v20 + 8);
      *(_QWORD *)(result + 8) = v19;
      v22 = a1[1];
      if ( (_DWORD)v22 )
        return sub_E98EB0(a2, v22, 0);
      return result;
    }
    result = sub_E990E0(a2, v23[0]);
  }
  v22 = a1[1];
  if ( (_DWORD)v22 )
    return sub_E98EB0(a2, v22, 0);
  return result;
}
