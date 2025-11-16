// Function: sub_21BF570
// Address: 0x21bf570
//
unsigned __int64 __fastcall sub_21BF570(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v11; // rbx
  int v12; // eax
  __int64 v13; // rcx
  _QWORD *v14; // rdi
  char v15; // si
  __int64 v16; // rcx
  __int64 v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 result; // rax
  __int128 v25; // [rsp-10h] [rbp-E0h]
  unsigned __int64 v26; // [rsp-10h] [rbp-E0h]
  __int64 v27; // [rsp+8h] [rbp-C8h]
  __int64 v28; // [rsp+10h] [rbp-C0h]
  unsigned int v29; // [rsp+18h] [rbp-B8h]
  unsigned int v30; // [rsp+1Ch] [rbp-B4h]
  __int64 v31; // [rsp+20h] [rbp-B0h] BYREF
  int v32; // [rsp+28h] [rbp-A8h]
  __int64 v33; // [rsp+30h] [rbp-A0h] BYREF
  int v34; // [rsp+38h] [rbp-98h]
  __int64 v35; // [rsp+40h] [rbp-90h]
  int v36; // [rsp+48h] [rbp-88h]
  __int64 v37; // [rsp+50h] [rbp-80h]
  int v38; // [rsp+58h] [rbp-78h]
  unsigned __int8 v39[8]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v40; // [rsp+68h] [rbp-68h]
  char v41; // [rsp+70h] [rbp-60h]
  __int64 v42; // [rsp+78h] [rbp-58h]
  char v43; // [rsp+80h] [rbp-50h]
  __int64 v44; // [rsp+88h] [rbp-48h]
  char v45; // [rsp+90h] [rbp-40h]
  __int64 v46; // [rsp+98h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 32);
  v5 = *(_QWORD *)(a2 + 72);
  v6 = *v4;
  v7 = *((unsigned int *)v4 + 2);
  v31 = v5;
  v8 = v4[5];
  v9 = *((unsigned int *)v4 + 12);
  v10 = v4[10];
  v11 = *((unsigned int *)v4 + 22);
  if ( v5 )
  {
    v27 = v4[5];
    v29 = *((_DWORD *)v4 + 12);
    v28 = v6;
    v30 = v7;
    sub_1623A60((__int64)&v31, v5, 2);
    v8 = v27;
    v9 = v29;
    v6 = v28;
    v7 = v30;
  }
  v12 = *(_DWORD *)(a2 + 64);
  v38 = v11;
  v36 = v7;
  v13 = *(_QWORD *)(v6 + 40) + 16 * v7;
  v34 = v9;
  v14 = *(_QWORD **)(a1 + 272);
  v32 = v12;
  v33 = v8;
  v35 = v6;
  v37 = v10;
  v40 = 0;
  v41 = 6;
  v42 = 0;
  v39[0] = 6;
  v15 = *(_BYTE *)v13;
  v16 = *(_QWORD *)(v13 + 8);
  v43 = v15;
  v44 = v16;
  v17 = *(_QWORD *)(v10 + 40) + 16 * v11;
  v18 = *(_QWORD *)(v17 + 8);
  *((_QWORD *)&v25 + 1) = 3;
  *(_QWORD *)&v25 = &v33;
  v45 = *(_BYTE *)v17;
  v46 = v18;
  v19 = sub_1D25E10(v14, 576, (__int64)&v31, v39, 4, v9, v25);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v19);
  sub_1D49010(v19);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v20, v21, v22, v23);
  result = v26;
  if ( v31 )
    return sub_161E7C0((__int64)&v31, v31);
  return result;
}
