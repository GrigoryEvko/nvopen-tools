// Function: sub_38B4C10
// Address: 0x38b4c10
//
__int64 __fastcall sub_38B4C10(__int64 a1, __int64 a2, int *a3, unsigned int a4)
{
  unsigned int v7; // r15d
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rax
  _BYTE *v12; // rsi
  char v13; // cl
  __int64 v14; // rdx
  int v15; // [rsp+4h] [rbp-ACh]
  unsigned __int64 v16; // [rsp+8h] [rbp-A8h]
  int v17; // [rsp+8h] [rbp-A8h]
  __int64 v18; // [rsp+10h] [rbp-A0h]
  int v19; // [rsp+24h] [rbp-8Ch] BYREF
  __int64 v20; // [rsp+28h] [rbp-88h] BYREF
  __m128i v21; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v22; // [rsp+40h] [rbp-70h] BYREF
  __int64 v23; // [rsp+48h] [rbp-68h]
  __int64 v24; // [rsp+50h] [rbp-60h]
  __int64 v25[2]; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+70h] [rbp-40h] BYREF

  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v21 = 0u;
  LOBYTE(v19) = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388F6D0(a1, &v21)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388F470(a1, &v19)
    || *(_DWORD *)(a1 + 64) == 4 && (*(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8), sub_38B4790(a1, (__int64)&v22))
    || (v7 = sub_388AF10(a1, 13, "expected ')' here"), (_BYTE)v7) )
  {
    v7 = 1;
  }
  else
  {
    v9 = v24;
    v24 = 0;
    v10 = v23;
    v23 = 0;
    v16 = v22;
    v15 = v19;
    v18 = v10;
    v22 = 0;
    v11 = sub_22077B0(0x40u);
    if ( v11 )
    {
      *(_QWORD *)(v11 + 40) = v16;
      *(_DWORD *)(v11 + 8) = 2;
      *(_QWORD *)(v11 + 48) = v18;
      *(_DWORD *)(v11 + 12) = v15;
      *(_QWORD *)(v11 + 16) = 0;
      *(_QWORD *)(v11 + 56) = v9;
      *(_QWORD *)v11 = &unk_49EB4D8;
    }
    else if ( v16 )
    {
      j_j___libc_free_0(v16);
      v11 = 0;
    }
    v12 = *(_BYTE **)a2;
    v20 = v11;
    v13 = v19;
    *(__m128i *)(v11 + 24) = v21;
    v14 = *(_QWORD *)(a2 + 8);
    v25[0] = (__int64)v26;
    v17 = v13 & 0xF;
    sub_3887850(v25, v12, (__int64)&v12[v14]);
    sub_3895460(a1, (__int64)v25, a3, v17, a4, &v20);
    if ( (_QWORD *)v25[0] != v26 )
      j_j___libc_free_0(v25[0]);
    if ( v20 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
  }
  if ( v22 )
    j_j___libc_free_0(v22);
  return v7;
}
