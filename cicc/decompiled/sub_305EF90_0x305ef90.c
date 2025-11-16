// Function: sub_305EF90
// Address: 0x305ef90
//
__int64 __fastcall sub_305EF90(__int64 a1, unsigned int a2, __int64 *a3, __int64 *a4)
{
  unsigned int v4; // r15d
  unsigned int v6; // r14d
  __int64 *v7; // rax
  unsigned int v8; // eax
  __int64 v9; // r12
  unsigned int v10; // r10d
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax
  char v14; // dl
  __int64 *v15; // rax
  unsigned __int16 v16; // r12
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v18; // ax
  __int64 v20; // rcx
  unsigned int v22; // [rsp+10h] [rbp-60h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  char v24[8]; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int16 v25; // [rsp+28h] [rbp-48h]

  v4 = a2;
  if ( a2 > 2 )
  {
    while ( 1 )
    {
      v6 = v4;
      v4 >>= 1;
      v7 = (__int64 *)sub_BCDA70(a3, v4);
      v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), v7, 0);
      v9 = *(_QWORD *)(a1 + 32);
      v10 = v8;
      v12 = v11;
      v13 = 1;
      if ( (_WORD)v10 == 1 )
        goto LABEL_5;
      if ( (_WORD)v10 )
        break;
LABEL_7:
      v22 = v10;
      v23 = v12;
      v15 = (__int64 *)sub_BCDA70(a4, v4);
      v16 = sub_2D5BAE0(v9, *(_QWORD *)(a1 + 16), v15, 0);
      v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**(_QWORD **)(a1 + 32) + 592LL);
      if ( v17 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)v24, *(_QWORD *)(a1 + 32), *a3, v22, v23);
        v18 = v25;
      }
      else
      {
        v18 = v17(*(_QWORD *)(a1 + 32), *a3, v22, v23);
      }
      if ( !v18 )
        return v6;
      v20 = *(_QWORD *)(a1 + 32);
      if ( !*(_QWORD *)(v20 + 8LL * v18 + 112) || !v16 || *(_BYTE *)(v16 + v20 + 274LL * v18 + 443718) )
        return v6;
LABEL_14:
      if ( v4 <= 2 )
        return v4;
    }
    v13 = (unsigned __int16)v10;
    v14 = *(_BYTE *)(v9 + 500LL * (unsigned __int16)v10 + 6713);
    if ( *(_QWORD *)(v9 + 8LL * (unsigned __int16)v10 + 112) )
    {
LABEL_5:
      v14 = *(_BYTE *)(v9 + 500 * v13 + 6713);
      if ( !v14 )
        goto LABEL_14;
    }
    if ( v14 == 4 )
      goto LABEL_14;
    goto LABEL_7;
  }
  return v4;
}
