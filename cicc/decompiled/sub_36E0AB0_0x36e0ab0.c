// Function: sub_36E0AB0
// Address: 0x36e0ab0
//
void __fastcall sub_36E0AB0(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // r10
  int v9; // r9d
  __int64 v10; // r15
  __int64 v11; // rbx
  int v12; // eax
  __int64 v13; // rcx
  _QWORD *v14; // rdi
  __int64 v15; // rbx
  __int16 v16; // si
  __int64 v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int128 v23; // [rsp-10h] [rbp-E0h]
  __int64 v24; // [rsp+8h] [rbp-C8h]
  __int64 v25; // [rsp+10h] [rbp-C0h]
  int v26; // [rsp+18h] [rbp-B8h]
  unsigned int v27; // [rsp+1Ch] [rbp-B4h]
  __int64 v28; // [rsp+20h] [rbp-B0h] BYREF
  int v29; // [rsp+28h] [rbp-A8h]
  __int64 v30; // [rsp+30h] [rbp-A0h] BYREF
  int v31; // [rsp+38h] [rbp-98h]
  __int64 v32; // [rsp+40h] [rbp-90h]
  int v33; // [rsp+48h] [rbp-88h]
  __int64 v34; // [rsp+50h] [rbp-80h]
  int v35; // [rsp+58h] [rbp-78h]
  unsigned __int16 v36; // [rsp+60h] [rbp-70h] BYREF
  __int64 v37; // [rsp+68h] [rbp-68h]
  __int16 v38; // [rsp+70h] [rbp-60h]
  __int64 v39; // [rsp+78h] [rbp-58h]
  __int16 v40; // [rsp+80h] [rbp-50h]
  __int64 v41; // [rsp+88h] [rbp-48h]
  __int16 v42; // [rsp+90h] [rbp-40h]
  __int64 v43; // [rsp+98h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = *((unsigned int *)v4 + 2);
  v28 = v5;
  v8 = v4[5];
  v9 = *((_DWORD *)v4 + 12);
  v10 = v4[10];
  v11 = *((unsigned int *)v4 + 22);
  if ( v5 )
  {
    v24 = v4[5];
    v26 = *((_DWORD *)v4 + 12);
    v25 = v6;
    v27 = v7;
    sub_B96E90((__int64)&v28, v5, 1);
    v8 = v24;
    v9 = v26;
    v6 = v25;
    v7 = v27;
  }
  v12 = *(_DWORD *)(a2 + 72);
  v33 = v7;
  v13 = *(_QWORD *)(v6 + 48) + 16 * v7;
  v34 = v10;
  v31 = v9;
  v14 = *(_QWORD **)(a1 + 64);
  v38 = 8;
  v30 = v8;
  v29 = v12;
  v32 = v6;
  v35 = v11;
  v15 = *(_QWORD *)(v10 + 48) + 16 * v11;
  v37 = 0;
  v39 = 0;
  v36 = 8;
  v16 = *(_WORD *)v13;
  v17 = *(_QWORD *)(v13 + 8);
  v40 = v16;
  v41 = v17;
  v18 = *(_QWORD *)(v15 + 8);
  *((_QWORD *)&v23 + 1) = 3;
  *(_QWORD *)&v23 = &v30;
  v42 = *(_WORD *)v15;
  v43 = v18;
  v19 = sub_33E6B60(v14, 1558, (__int64)&v28, &v36, 4, 8, v23);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v19, v20, v21, v22);
  sub_3421DB0(v19);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
}
