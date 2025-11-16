// Function: sub_108BD50
// Address: 0x108bd50
//
__int64 __fastcall sub_108BD50(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r10d
  __int64 v5; // r14
  __int16 v6; // si
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdi
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 *v15; // rbx
  __int64 i; // r15
  __int64 v17; // rsi
  __int64 *v18; // rdi
  __int64 *v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+10h] [rbp-A0h]
  int v25; // [rsp+1Ch] [rbp-94h]
  __int64 v26; // [rsp+20h] [rbp-90h]
  __int64 *v27; // [rsp+28h] [rbp-88h]
  __int64 v28; // [rsp+28h] [rbp-88h]
  __int64 v29; // [rsp+30h] [rbp-80h]

  v4 = *(__int16 *)(a2 + 56);
  v5 = *(_QWORD *)(a1 + 1824);
  v6 = *(_WORD *)(a1 + 152) + 1;
  *(_WORD *)(a1 + 152) = v6;
  if ( v5 != *(_QWORD *)(a1 + 1832) )
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = off_497C020;
      *(_QWORD *)(v5 + 16) = a3;
      *(_QWORD *)(v5 + 8) = 0x6F6C6672766F2ELL;
      *(_QWORD *)(v5 + 24) = 0;
      *(_QWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      *(_DWORD *)(v5 + 48) = v4;
      *(_DWORD *)(v5 + 52) = 0x8000;
      *(_WORD *)(v5 + 56) = v6;
      v5 = *(_QWORD *)(a1 + 1824);
    }
    *(_QWORD *)(a1 + 1824) = v5 + 64;
    goto LABEL_5;
  }
  v8 = v5 - *(_QWORD *)(a1 + 1816);
  v27 = *(__int64 **)(a1 + 1816);
  v9 = v8 >> 6;
  if ( v8 >> 6 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = v8 >> 6;
  v11 = __CFADD__(v10, v9);
  v12 = v10 + v9;
  if ( v11 )
  {
    v20 = 0x7FFFFFFFFFFFFFC0LL;
LABEL_27:
    v23 = a3;
    v24 = v8;
    v25 = v4;
    v21 = sub_22077B0(v20);
    v4 = v25;
    v22 = v21 + v20;
    v29 = v21;
    v8 = v24;
    v13 = v21 + 64;
    v26 = v22;
    a3 = v23;
    goto LABEL_12;
  }
  if ( v12 )
  {
    if ( v12 > 0x1FFFFFFFFFFFFFFLL )
      v12 = 0x1FFFFFFFFFFFFFFLL;
    v20 = v12 << 6;
    goto LABEL_27;
  }
  v26 = 0;
  v13 = 64;
  v29 = 0;
LABEL_12:
  v14 = v29 + v8;
  if ( v14 )
  {
    *(_QWORD *)v14 = off_497C020;
    *(_QWORD *)(v14 + 16) = a3;
    *(_QWORD *)(v14 + 8) = 0x6F6C6672766F2ELL;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 40) = 0;
    *(_DWORD *)(v14 + 48) = v4;
    *(_DWORD *)(v14 + 52) = 0x8000;
    *(_WORD *)(v14 + 56) = v6;
  }
  v15 = v27;
  if ( (__int64 *)v5 != v27 )
  {
    for ( i = v29; ; i += 64 )
    {
      if ( i )
      {
        *(_QWORD *)i = off_497C020;
        *(_QWORD *)(i + 8) = v15[1];
        *(_QWORD *)(i + 16) = v15[2];
        *(_QWORD *)(i + 24) = v15[3];
        *(_QWORD *)(i + 32) = v15[4];
        *(_QWORD *)(i + 40) = v15[5];
        *(_DWORD *)(i + 48) = *((_DWORD *)v15 + 12);
        *(_DWORD *)(i + 52) = *((_DWORD *)v15 + 13);
        *(_WORD *)(i + 56) = *((_WORD *)v15 + 28);
      }
      v17 = *v15;
      v18 = v15;
      v15 += 8;
      (*(void (__fastcall **)(__int64 *))(v17 + 16))(v18);
      if ( (__int64 *)v5 == v15 )
        break;
    }
    v13 = i + 128;
  }
  v19 = v27;
  if ( v27 )
  {
    v28 = v13;
    j_j___libc_free_0(v19, *(_QWORD *)(a1 + 1832) - (_QWORD)v19);
    v13 = v28;
  }
  *(_QWORD *)(a1 + 1824) = v13;
  *(_QWORD *)(a1 + 1816) = v29;
  *(_QWORD *)(a1 + 1832) = v26;
LABEL_5:
  *(_DWORD *)(a2 + 48) = 0xFFFF;
  return a2;
}
