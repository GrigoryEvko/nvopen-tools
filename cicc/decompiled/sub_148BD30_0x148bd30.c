// Function: sub_148BD30
// Address: 0x148bd30
//
__int64 __fastcall sub_148BD30(_QWORD **a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // r15
  __int64 v6; // rax
  _QWORD *v7; // rdi
  unsigned int v8; // esi
  __int64 *v9; // rdx
  __int64 v10; // r10
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
  _QWORD *v32; // [rsp+10h] [rbp-90h]
  _QWORD *v33; // [rsp+10h] [rbp-90h]
  _QWORD *v34; // [rsp+10h] [rbp-90h]
  _QWORD *v35; // [rsp+10h] [rbp-90h]
  char v36; // [rsp+1Fh] [rbp-81h]
  char v37; // [rsp+1Fh] [rbp-81h]
  char v38; // [rsp+1Fh] [rbp-81h]
  char v39; // [rsp+1Fh] [rbp-81h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+28h] [rbp-78h] BYREF
  __int64 v45; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v46; // [rsp+40h] [rbp-60h] BYREF
  __int64 v47; // [rsp+48h] [rbp-58h]
  _QWORD v48[10]; // [rsp+50h] [rbp-50h] BYREF

  v4 = a2;
  v6 = *((unsigned int *)a1 + 8);
  v44 = a2;
  if ( (_DWORD)v6 )
  {
    v7 = a1[2];
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
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
      v30 = sub_148BD30(a1, *(_QWORD *)(v4 + 32));
      if ( v30 != *(_QWORD *)(v4 + 32) )
        v4 = sub_14835F0(*a1, v30, *(_QWORD *)(v4 + 40), 0, a3, a4);
      break;
    case 2:
      v22 = sub_148BD30(a1, *(_QWORD *)(v4 + 32));
      if ( v22 != *(_QWORD *)(v4 + 32) )
        v4 = sub_14747F0((__int64)*a1, v22, *(_QWORD *)(v4 + 40), 0);
      break;
    case 3:
      v26 = sub_148BD30(a1, *(_QWORD *)(v4 + 32));
      if ( v26 != *(_QWORD *)(v4 + 32) )
        v4 = sub_147B0D0((__int64)*a1, v26, *(_QWORD *)(v4 + 40), 0);
      break;
    case 4:
      v46 = v48;
      v47 = 0x200000000LL;
      v23 = *(_QWORD **)(v4 + 32);
      v34 = &v23[*(_QWORD *)(v4 + 40)];
      if ( v23 != v34 )
      {
        v38 = 0;
        v24 = *(_QWORD **)(v4 + 32);
        do
        {
          v42 = *v24;
          v45 = sub_148BD30(a1, *v24);
          sub_1458920((__int64)&v46, &v45);
          v15 = v46;
          ++v24;
          v38 |= v46[(unsigned int)v47 - 1] != v42;
        }
        while ( v34 != v24 );
        if ( v38 )
        {
          v25 = sub_147DD40((__int64)*a1, (__int64 *)&v46, 0, 0, a3, a4);
          v15 = v46;
          v4 = (__int64)v25;
        }
        goto LABEL_39;
      }
      break;
    case 5:
      v46 = v48;
      v47 = 0x200000000LL;
      v27 = *(_QWORD **)(v4 + 32);
      v35 = &v27[*(_QWORD *)(v4 + 40)];
      if ( v27 != v35 )
      {
        v39 = 0;
        v28 = *(_QWORD **)(v4 + 32);
        do
        {
          v43 = *v28;
          v45 = sub_148BD30(a1, *v28);
          sub_1458920((__int64)&v46, &v45);
          v15 = v46;
          ++v28;
          v39 |= v46[(unsigned int)v47 - 1] != v43;
        }
        while ( v35 != v28 );
        if ( v39 )
        {
          v29 = sub_147EE30(*a1, &v46, 0, 0, a3, a4);
          v15 = v46;
          v4 = v29;
        }
        goto LABEL_39;
      }
      break;
    case 6:
      v20 = sub_148BD30(a1, *(_QWORD *)(v4 + 32));
      v21 = sub_148BD30(a1, *(_QWORD *)(v4 + 40));
      if ( v20 != *(_QWORD *)(v4 + 32) || v21 != *(_QWORD *)(v4 + 40) )
        v4 = sub_1483CF0(*a1, v20, v21, a3, a4);
      break;
    case 7:
      if ( *(_QWORD **)(v4 + 48) == a1[5] )
        v4 = **(_QWORD **)(v4 + 32);
      else
        *((_BYTE *)a1 + 49) = 1;
      break;
    case 8:
      v46 = v48;
      v47 = 0x200000000LL;
      v17 = *(_QWORD **)(v4 + 32);
      v33 = &v17[*(_QWORD *)(v4 + 40)];
      if ( v17 != v33 )
      {
        v37 = 0;
        v18 = *(_QWORD **)(v4 + 32);
        do
        {
          v41 = *v18;
          v45 = sub_148BD30(a1, *v18);
          sub_1458920((__int64)&v46, &v45);
          v15 = v46;
          ++v18;
          v37 |= v46[(unsigned int)v47 - 1] != v41;
        }
        while ( v33 != v18 );
        if ( v37 )
        {
          v19 = sub_14813B0(*a1, &v46, a3, a4);
          v15 = v46;
          v4 = v19;
        }
        goto LABEL_39;
      }
      break;
    case 9:
      v46 = v48;
      v47 = 0x200000000LL;
      v13 = *(_QWORD **)(v4 + 32);
      v32 = &v13[*(_QWORD *)(v4 + 40)];
      if ( v13 != v32 )
      {
        v36 = 0;
        v14 = *(_QWORD **)(v4 + 32);
        do
        {
          v40 = *v14;
          v45 = sub_148BD30(a1, *v14);
          sub_1458920((__int64)&v46, &v45);
          v15 = v46;
          ++v14;
          v36 |= v46[(unsigned int)v47 - 1] != v40;
        }
        while ( v32 != v14 );
        if ( v36 )
        {
          v16 = sub_147A3C0(*a1, &v46, a3, a4);
          v15 = v46;
          v4 = v16;
        }
LABEL_39:
        if ( v15 != v48 )
          _libc_free((unsigned __int64)v15);
      }
      break;
    case 0xA:
      if ( !sub_146CEE0((__int64)*a1, v4, (__int64)a1[5]) )
        *((_BYTE *)a1 + 48) = 1;
      break;
  }
  v45 = v4;
  sub_1466830((__int64)&v46, (__int64)(a1 + 1), &v44, &v45);
  return *(_QWORD *)(v48[0] + 8LL);
}
