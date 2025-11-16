// Function: sub_38EF860
// Address: 0x38ef860
//
__int64 __fastcall sub_38EF860(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // edx
  _BYTE *v13; // rsi
  __int64 *v14; // rbx
  _QWORD *v15; // rdi
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rbx
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 result; // rax
  __int64 v21; // [rsp+0h] [rbp-60h] BYREF
  __int64 v22; // [rsp+8h] [rbp-58h] BYREF
  __int64 v23[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h]

  v5 = a3[3];
  if ( (unsigned __int64)(a3[2] - v5) <= 5 )
  {
    sub_16E7EE0((__int64)a3, ".endr\n", 6u);
  }
  else
  {
    *(_DWORD *)v5 = 1684956462;
    *(_WORD *)(v5 + 4) = 2674;
    a3[3] += 6LL;
  }
  v23[0] = (__int64)"<instantiation>";
  v6 = a3[5];
  LOWORD(v24) = 259;
  sub_16C28C0(&v21, *(const void **)v6, *(unsigned int *)(v6 + 8), (__int64)v23);
  v7 = sub_3909460(a1);
  v8 = sub_39092A0(v7);
  v9 = (__int64)(*(_QWORD *)(a1 + 400) - *(_QWORD *)(a1 + 392)) >> 3;
  v10 = v8;
  v11 = sub_22077B0(0x20u);
  if ( v11 )
  {
    v12 = *(_DWORD *)(a1 + 376);
    *(_QWORD *)v11 = a2;
    *(_QWORD *)(v11 + 16) = v10;
    *(_DWORD *)(v11 + 8) = v12;
    *(_QWORD *)(v11 + 24) = v9;
  }
  v22 = v11;
  v13 = *(_BYTE **)(a1 + 456);
  if ( v13 == *(_BYTE **)(a1 + 464) )
  {
    sub_38E2F40(a1 + 448, v13, &v22);
  }
  else
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = v11;
      v13 = *(_BYTE **)(a1 + 456);
    }
    *(_QWORD *)(a1 + 456) = v13 + 8;
  }
  v14 = *(__int64 **)(a1 + 344);
  v23[1] = 0;
  v24 = 0;
  v23[0] = v21;
  v15 = (_QWORD *)v14[1];
  v21 = 0;
  if ( v15 == (_QWORD *)v14[2] )
  {
    sub_168C7C0(v14, (__int64)v15, (__int64)v23);
    v16 = (_QWORD *)v14[1];
  }
  else
  {
    if ( v15 )
    {
      sub_16CE2D0(v15, v23);
      v15 = (_QWORD *)v14[1];
    }
    v16 = v15 + 3;
    v14[1] = (__int64)v16;
  }
  v17 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v16 - *v14) >> 3);
  sub_16CE300(v23);
  v18 = *(_QWORD **)(a1 + 344);
  *(_DWORD *)(a1 + 376) = v17;
  v19 = *(_QWORD *)(*v18 + 24LL * (unsigned int)(v17 - 1));
  sub_392A730(a1 + 144, *(_QWORD *)(v19 + 8), *(_QWORD *)(v19 + 16) - *(_QWORD *)(v19 + 8), 0);
  result = sub_38EB180(a1);
  if ( v21 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
  return result;
}
