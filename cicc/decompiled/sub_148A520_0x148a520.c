// Function: sub_148A520
// Address: 0x148a520
//
__int64 __fastcall sub_148A520(__int64 *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // r15
  __int64 v6; // rax
  _QWORD *v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  int v12; // edx
  _QWORD *v13; // rdx
  _QWORD *v14; // rbx
  __int64 *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rdx
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rsi
  _QWORD *v23; // rdx
  _QWORD *v24; // rbx
  __int64 *v25; // rax
  __int64 v26; // rsi
  _QWORD *v27; // rdx
  _QWORD *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // r11d
  __int64 v32; // r12
  __int64 v33; // rax
  _QWORD *v34; // [rsp+10h] [rbp-90h]
  _QWORD *v35; // [rsp+10h] [rbp-90h]
  _QWORD *v36; // [rsp+10h] [rbp-90h]
  _QWORD *v37; // [rsp+10h] [rbp-90h]
  char v38; // [rsp+1Fh] [rbp-81h]
  char v39; // [rsp+1Fh] [rbp-81h]
  char v40; // [rsp+1Fh] [rbp-81h]
  char v41; // [rsp+1Fh] [rbp-81h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+20h] [rbp-80h]
  __int64 v46; // [rsp+28h] [rbp-78h] BYREF
  __int64 v47; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v48; // [rsp+40h] [rbp-60h] BYREF
  __int64 v49; // [rsp+48h] [rbp-58h]
  _QWORD v50[10]; // [rsp+50h] [rbp-50h] BYREF

  v4 = a2;
  v6 = *((unsigned int *)a1 + 8);
  v46 = a2;
  if ( (_DWORD)v6 )
  {
    v7 = (_QWORD *)a1[2];
    v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v9 = &v7[2 * v8];
    v10 = *v9;
    if ( v4 == *v9 )
    {
LABEL_3:
      if ( v9 != &v7[2 * v6] )
        return v9[1];
    }
    else
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v31 = v12 + 1;
        v8 = (v6 - 1) & (v12 + v8);
        v9 = &v7[2 * v8];
        v10 = *v9;
        if ( v4 == *v9 )
          goto LABEL_3;
        v12 = v31;
      }
    }
  }
  switch ( *(_WORD *)(v4 + 24) )
  {
    case 0:
    case 0xB:
      break;
    case 1:
      v30 = sub_148A520(a1, *(_QWORD *)(v4 + 32));
      if ( v30 != *(_QWORD *)(v4 + 32) )
        v4 = sub_14835F0((_QWORD *)*a1, v30, *(_QWORD *)(v4 + 40), 0, a3, a4);
      break;
    case 2:
      v22 = sub_148A520(a1, *(_QWORD *)(v4 + 32));
      if ( v22 != *(_QWORD *)(v4 + 32) )
        v4 = sub_14747F0(*a1, v22, *(_QWORD *)(v4 + 40), 0);
      break;
    case 3:
      v26 = sub_148A520(a1, *(_QWORD *)(v4 + 32));
      if ( v26 != *(_QWORD *)(v4 + 32) )
        v4 = sub_147B0D0(*a1, v26, *(_QWORD *)(v4 + 40), 0);
      break;
    case 4:
      v48 = v50;
      v49 = 0x200000000LL;
      v23 = *(_QWORD **)(v4 + 32);
      v36 = &v23[*(_QWORD *)(v4 + 40)];
      if ( v23 != v36 )
      {
        v40 = 0;
        v24 = *(_QWORD **)(v4 + 32);
        do
        {
          v44 = *v24;
          v47 = sub_148A520(a1, *v24);
          sub_1458920((__int64)&v48, &v47);
          v15 = v48;
          ++v24;
          v40 |= v48[(unsigned int)v49 - 1] != v44;
        }
        while ( v36 != v24 );
        if ( v40 )
        {
          v25 = sub_147DD40(*a1, (__int64 *)&v48, 0, 0, a3, a4);
          v15 = v48;
          v4 = (__int64)v25;
        }
        goto LABEL_37;
      }
      break;
    case 5:
      v48 = v50;
      v49 = 0x200000000LL;
      v27 = *(_QWORD **)(v4 + 32);
      v37 = &v27[*(_QWORD *)(v4 + 40)];
      if ( v27 != v37 )
      {
        v41 = 0;
        v28 = *(_QWORD **)(v4 + 32);
        do
        {
          v45 = *v28;
          v47 = sub_148A520(a1, *v28);
          sub_1458920((__int64)&v48, &v47);
          v15 = v48;
          ++v28;
          v41 |= v48[(unsigned int)v49 - 1] != v45;
        }
        while ( v37 != v28 );
        if ( v41 )
        {
          v29 = sub_147EE30((_QWORD *)*a1, &v48, 0, 0, a3, a4);
          v15 = v48;
          v4 = v29;
        }
        goto LABEL_37;
      }
      break;
    case 6:
      v20 = sub_148A520(a1, *(_QWORD *)(v4 + 32));
      v21 = sub_148A520(a1, *(_QWORD *)(v4 + 40));
      if ( v20 != *(_QWORD *)(v4 + 32) || v21 != *(_QWORD *)(v4 + 40) )
        v4 = sub_1483CF0((_QWORD *)*a1, v20, v21, a3, a4);
      break;
    case 7:
      if ( *(_QWORD *)(v4 + 48) != a1[5] || *(_QWORD *)(v4 + 40) != 2 )
        goto LABEL_9;
      v32 = *a1;
      v33 = sub_13A5BC0((_QWORD *)v4, *a1);
      v4 = sub_14806B0(v32, v4, v33, 0, 0);
      break;
    case 8:
      v48 = v50;
      v49 = 0x200000000LL;
      v17 = *(_QWORD **)(v4 + 32);
      v35 = &v17[*(_QWORD *)(v4 + 40)];
      if ( v17 != v35 )
      {
        v39 = 0;
        v18 = *(_QWORD **)(v4 + 32);
        do
        {
          v43 = *v18;
          v47 = sub_148A520(a1, *v18);
          sub_1458920((__int64)&v48, &v47);
          v15 = v48;
          ++v18;
          v39 |= v48[(unsigned int)v49 - 1] != v43;
        }
        while ( v35 != v18 );
        if ( v39 )
        {
          v19 = sub_14813B0((_QWORD *)*a1, &v48, a3, a4);
          v15 = v48;
          v4 = v19;
        }
        goto LABEL_37;
      }
      break;
    case 9:
      v48 = v50;
      v49 = 0x200000000LL;
      v13 = *(_QWORD **)(v4 + 32);
      v34 = &v13[*(_QWORD *)(v4 + 40)];
      if ( v13 != v34 )
      {
        v38 = 0;
        v14 = *(_QWORD **)(v4 + 32);
        do
        {
          v42 = *v14;
          v47 = sub_148A520(a1, *v14);
          sub_1458920((__int64)&v48, &v47);
          v15 = v48;
          ++v14;
          v38 |= v48[(unsigned int)v49 - 1] != v42;
        }
        while ( v34 != v14 );
        if ( v38 )
        {
          v16 = sub_147A3C0((_QWORD *)*a1, &v48, a3, a4);
          v15 = v48;
          v4 = v16;
        }
LABEL_37:
        if ( v15 != v50 )
          _libc_free((unsigned __int64)v15);
      }
      break;
    case 0xA:
      if ( !sub_146CEE0(*a1, v4, a1[5]) )
LABEL_9:
        *((_BYTE *)a1 + 48) = 0;
      break;
  }
  v47 = v4;
  sub_1466830((__int64)&v48, (__int64)(a1 + 1), &v46, &v47);
  return *(_QWORD *)(v50[0] + 8LL);
}
