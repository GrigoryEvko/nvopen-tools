// Function: sub_330A840
// Address: 0x330a840
//
__int64 __fastcall sub_330A840(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rbx
  unsigned int v10; // r13d
  __int128 v11; // xmm1
  unsigned __int16 *v12; // rax
  __int64 v13; // rcx
  int v14; // r14d
  __int64 v15; // rbx
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rax
  int v20; // edx
  int v21; // ebx
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // rax
  int v26; // edx
  int v27; // ebx
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rax
  int v32; // edx
  int v33; // [rsp+Ch] [rbp-94h]
  int v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int128 v36; // [rsp+20h] [rbp-80h]
  __int64 v37; // [rsp+40h] [rbp-60h] BYREF
  int v38; // [rsp+48h] [rbp-58h]
  __int64 v39; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-48h]
  __int64 v41; // [rsp+60h] [rbp-40h]
  int v42; // [rsp+68h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(_QWORD *)v7;
  v10 = *(_DWORD *)(v7 + 8);
  v11 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v36 = (__int128)_mm_loadu_si128((const __m128i *)v7);
  v35 = *(_QWORD *)(v7 + 40);
  v33 = *(_DWORD *)(v7 + 48);
  v12 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * v10);
  v13 = *((_QWORD *)v12 + 1);
  v14 = *v12;
  v37 = v8;
  v34 = v13;
  if ( v8 )
    sub_B96E90((__int64)&v37, v8, 1);
  v38 = *(_DWORD *)(a2 + 72);
  if ( !(unsigned __int8)sub_33CF8A0(a2, 1, a3, v13, a5, a6) )
  {
    v19 = sub_33F17F0(*a1, 67, &v37, 262, 0);
    v21 = v20;
    v22 = v19;
    v23 = sub_3406EB0(*a1, 57, (unsigned int)&v37, v14, v34, (unsigned int)&v37, v36, v11);
    goto LABEL_14;
  }
  if ( v35 != v9 || v33 != v10 )
  {
    if ( (unsigned __int8)sub_33CF170(v11, *((_QWORD *)&v11 + 1)) )
    {
      v17 = sub_33F17F0(*a1, 67, &v37, 262, 0);
      v39 = v9;
      v40 = v10;
      v41 = v17;
      v42 = v18;
LABEL_12:
      v15 = sub_32EB790((__int64)a1, a2, &v39, 2, 1);
      goto LABEL_8;
    }
    v15 = 0;
    if ( !(unsigned __int8)sub_33CF460(v36, *((_QWORD *)&v36 + 1)) )
      goto LABEL_8;
    v31 = sub_33F17F0(*a1, 67, &v37, 262, 0);
    v21 = v32;
    v22 = v31;
    v23 = sub_3406EB0(*a1, 188, (unsigned int)&v37, v14, v34, (unsigned int)&v37, v11, v36);
LABEL_14:
    v41 = v22;
    v39 = v23;
    v40 = v24;
    v42 = v21;
    goto LABEL_12;
  }
  v25 = sub_33F17F0(*a1, 67, &v37, 262, 0);
  v27 = v26;
  v28 = v25;
  v29 = sub_3400BD0(*a1, 0, (unsigned int)&v37, v14, v34, 0, 0);
  v40 = v30;
  v41 = v28;
  v42 = v27;
  v39 = v29;
  v15 = sub_32EB790((__int64)a1, a2, &v39, 2, 1);
LABEL_8:
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  return v15;
}
