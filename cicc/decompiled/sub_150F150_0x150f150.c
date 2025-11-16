// Function: sub_150F150
// Address: 0x150f150
//
__int64 __fastcall sub_150F150(__int64 a1, __int64 a2, __int64 a3, char a4, __m128i a5)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 (*v13)(void); // rax
  unsigned __int64 v15; // rax
  char v16; // al
  _QWORD v17[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v18[2]; // [rsp+10h] [rbp-60h] BYREF
  char v19; // [rsp+20h] [rbp-50h]
  __m128i *v20; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v21; // [rsp+38h] [rbp-38h]
  char *v22; // [rsp+40h] [rbp-30h]
  __int64 v23; // [rsp+48h] [rbp-28h]

  v17[0] = a2;
  v17[1] = a3;
  LOWORD(v22) = 261;
  v20 = (__m128i *)v17;
  sub_16C2E90(v18, &v20, -1, 1);
  if ( (v19 & 1) != 0 )
  {
    sub_16BCB40(&v20, LODWORD(v18[0]), v18[1]);
    v15 = (unsigned __int64)v20;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v15 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v19 & 1) != 0 )
      return a1;
    goto LABEL_9;
  }
  v10 = v18[0];
  v11 = *(_QWORD *)(v18[0] + 16LL);
  v12 = *(_QWORD *)(v18[0] + 8LL);
  if ( !a4 || v11 != v12 )
  {
    v20 = *(__m128i **)(v18[0] + 8LL);
    v21 = v11 - v12;
    v13 = *(__int64 (**)(void))(*(_QWORD *)v18[0] + 16LL);
    if ( (char *)v13 == (char *)sub_12BCB10 )
    {
      v23 = 14;
      v22 = "Unknown buffer";
    }
    else
    {
      v22 = (char *)v13();
      v23 = v12;
    }
    sub_150F0A0(a1, (__int64)&v20, v12, v7, v8, v9, a5, v20, v21);
    if ( (v19 & 1) != 0 )
      return a1;
LABEL_9:
    v10 = v18[0];
    if ( v18[0] )
      goto LABEL_12;
    return a1;
  }
  v16 = *(_BYTE *)(a1 + 8);
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 8) = v16 & 0xFC | 2;
LABEL_12:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
  return a1;
}
