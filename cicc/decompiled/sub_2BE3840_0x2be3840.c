// Function: sub_2BE3840
// Address: 0x2be3840
//
__int64 __fastcall sub_2BE3840(__int64 a1)
{
  char *v2; // rbx
  char *v3; // r12
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  __int64 v7; // r14
  unsigned __int64 v8; // rbx
  char *v9; // rax
  _BYTE *v10; // rsi
  char *v11; // rdi
  char *v12; // rax
  char *v13; // rdx
  char v14; // r15
  unsigned __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 result; // rax
  _BYTE *v19; // r10
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  unsigned __int64 v23; // r10
  __int64 v24; // r13
  __int64 v25; // r13
  __int64 v26; // r15
  __int64 v27; // [rsp+8h] [rbp-A8h]
  _QWORD *v28; // [rsp+10h] [rbp-A0h]
  __int64 v29; // [rsp+10h] [rbp-A0h]
  _QWORD *v30; // [rsp+18h] [rbp-98h]
  _BYTE *v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+20h] [rbp-90h]
  char v33; // [rsp+20h] [rbp-90h]
  unsigned __int64 v34[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v35; // [rsp+50h] [rbp-60h] BYREF
  __int64 v36[2]; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v37[8]; // [rsp+70h] [rbp-40h] BYREF

  v2 = *(char **)(a1 + 8);
  v3 = *(char **)a1;
  v4 = v2;
  if ( *(char **)a1 != v2 )
  {
    _BitScanReverse64(&v5, v2 - v3);
    sub_2BDC070(*(_QWORD *)a1, v2, 2LL * (int)(63 - (v5 ^ 0x3F)));
    sub_2BDBF50(v3, v2);
    v2 = *(char **)a1;
    v4 = *(char **)(a1 + 8);
  }
  v6 = v2;
  v7 = a1;
  v8 = 0;
  v9 = sub_2BDBC30(v6, v4);
  sub_2BE3780(a1, v9, *(char **)(a1 + 8));
  do
  {
    v10 = *(_BYTE **)(v7 + 8);
    v11 = *(char **)v7;
    LOBYTE(v36[0]) = v8;
    if ( sub_2BE37E0(v11, v10, (char *)v36) )
      goto LABEL_10;
    v12 = *(char **)(v7 + 48);
    v13 = *(char **)(v7 + 56);
    if ( v12 != v13 )
    {
      while ( *v12 > (char)v8 || v12[1] < (char)v8 )
      {
        v12 += 2;
        if ( v13 == v12 )
          goto LABEL_16;
      }
      goto LABEL_10;
    }
LABEL_16:
    v14 = sub_2BDBFE0(*(_QWORD **)(v7 + 104), (unsigned int)(char)v8, *(_WORD *)(v7 + 96), *(_BYTE *)(v7 + 98));
    if ( v14 )
      goto LABEL_10;
    v28 = *(_QWORD **)(v7 + 104);
    v32 = *(_QWORD *)(v7 + 32);
    v30 = (_QWORD *)sub_222F790(v28, (unsigned int)(char)v8);
    v19 = (_BYTE *)sub_22077B0(1u);
    *v19 = v8;
    v20 = *v30;
    v21 = v30;
    v22 = (__int64)v19;
    v27 = (__int64)(v19 + 1);
    v31 = v19;
    (*(void (__fastcall **)(_QWORD *, _BYTE *))(v20 + 40))(v21, v19);
    v29 = sub_221F880(v28, v22);
    v36[0] = (__int64)v37;
    sub_2BDC2F0(v36, v31, v27);
    (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64, __int64))(*(_QWORD *)v29 + 24LL))(
      v34,
      v29,
      v36[0],
      v36[0] + v36[1]);
    v23 = (unsigned __int64)v31;
    if ( (_QWORD *)v36[0] != v37 )
    {
      j_j___libc_free_0(v36[0]);
      v23 = (unsigned __int64)v31;
    }
    j_j___libc_free_0(v23);
    v24 = sub_2BDD0F0(*(_QWORD *)(v7 + 24), *(_QWORD *)(v7 + 32), (__int64)v34);
    if ( (__int64 *)v34[0] != &v35 )
      j_j___libc_free_0(v34[0]);
    if ( v32 != v24 )
      goto LABEL_10;
    v25 = *(_QWORD *)(v7 + 72);
    if ( v25 != *(_QWORD *)(v7 + 80) )
    {
      v33 = v14;
      v26 = *(_QWORD *)(v7 + 80);
      while ( sub_2BDBFE0(*(_QWORD **)(v7 + 104), (unsigned int)(char)v8, *(_WORD *)v25, *(_BYTE *)(v25 + 2)) )
      {
        v25 += 4;
        if ( v26 == v25 )
        {
          v14 = v33;
          goto LABEL_11;
        }
      }
LABEL_10:
      v14 = 1;
    }
LABEL_11:
    v15 = v8 >> 6;
    v16 = *(_QWORD *)(v7 + 8 * (v8 >> 6) + 120);
    v17 = 1LL << v8;
    if ( *(_BYTE *)(v7 + 112) == v14 )
      result = v16 & ~v17;
    else
      result = v16 | v17;
    ++v8;
    *(_QWORD *)(v7 + 8 * v15 + 120) = result;
  }
  while ( v8 != 256 );
  return result;
}
