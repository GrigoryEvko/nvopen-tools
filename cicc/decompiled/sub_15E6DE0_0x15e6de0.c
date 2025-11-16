// Function: sub_15E6DE0
// Address: 0x15e6de0
//
_QWORD *__fastcall sub_15E6DE0(__int64 a1, int a2, __int64 a3, __int64 *a4, int a5, __int64 a6, __int64 a7, __int64 a8)
{
  __int64 v10; // rcx
  int v11; // r8d
  int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // rcx
  _QWORD *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v32; // rax
  __int64 v33; // [rsp-8h] [rbp-88h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  unsigned int v38; // [rsp+20h] [rbp-60h]
  int v39; // [rsp+20h] [rbp-60h]
  unsigned int v40; // [rsp+20h] [rbp-60h]
  _QWORD v44[7]; // [rsp+48h] [rbp-38h] BYREF

  v10 = a7 + 56 * a8;
  if ( v10 == a7 )
  {
    v36 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    v40 = a3 + 1;
    v32 = sub_1648AB0(72, (unsigned int)(a3 + 1), (unsigned int)(16 * a8));
    v17 = v40;
    v18 = a8;
    v19 = v36;
    v25 = a3;
    v21 = (_QWORD *)v32;
    if ( v32 )
    {
LABEL_8:
      v37 = v18;
      v39 = v19;
      sub_15F1EA0(v21, **(_QWORD **)(v19 + 16), 54, &v21[-3 * v25 - 3], v17, 0);
      v21[7] = 0;
      sub_15F5B40((_DWORD)v21, v39, a1, a2, a3, a5, a7, v37);
      v16 = v33;
    }
  }
  else
  {
    v11 = a3;
    v12 = 0;
    v13 = a7;
    do
    {
      v14 = *(_QWORD *)(v13 + 40) - *(_QWORD *)(v13 + 32);
      v13 += 56;
      v12 += v14 >> 3;
    }
    while ( v10 != v13 );
    v35 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    v38 = v11 + 1;
    v15 = sub_1648AB0(72, (unsigned int)(v11 + 1 + v12), (unsigned int)(16 * a8));
    v17 = v38;
    v18 = a8;
    v19 = v35;
    v20 = a7 + 56 * a8;
    v21 = (_QWORD *)v15;
    if ( v15 )
    {
      v22 = a7;
      LODWORD(v23) = 0;
      do
      {
        v24 = *(_QWORD *)(v22 + 40) - *(_QWORD *)(v22 + 32);
        v22 += 56;
        v23 = (unsigned int)(v24 >> 3) + (unsigned int)v23;
      }
      while ( a7 + 56 * a8 != v22 );
      v17 = (unsigned int)v23 + v38;
      v25 = a3 + v23;
      goto LABEL_8;
    }
  }
  if ( a6 )
    sub_15F2500(v21, a6, v16, v20, v17);
  v26 = (unsigned __int64 *)a4[2];
  sub_157E9D0(a4[1] + 40, (__int64)v21);
  v27 = *v26;
  v28 = v21[3];
  v21[4] = v26;
  v27 &= 0xFFFFFFFFFFFFFFF8LL;
  v21[3] = v27 | v28 & 7;
  *(_QWORD *)(v27 + 8) = v21 + 3;
  *v26 = *v26 & 7 | (unsigned __int64)(v21 + 3);
  v29 = *a4;
  if ( *a4 )
  {
    v44[0] = *a4;
    sub_1623A60(v44, v29, 2);
    if ( v21[6] )
      sub_161E7C0(v21 + 6);
    v30 = v44[0];
    v21[6] = v44[0];
    if ( v30 )
      sub_1623210(v44, v30, v21 + 6);
  }
  return v21;
}
