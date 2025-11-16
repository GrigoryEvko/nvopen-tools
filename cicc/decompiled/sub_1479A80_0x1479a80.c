// Function: sub_1479A80
// Address: 0x1479a80
//
__int64 __fastcall sub_1479A80(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r10
  int v10; // edx
  _QWORD *v11; // rdx
  _QWORD *v12; // rbx
  _QWORD *v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rdx
  _QWORD *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rsi
  _QWORD *v21; // rdx
  _QWORD *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rsi
  _QWORD *v25; // rdx
  _QWORD *v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rsi
  int v29; // r11d
  _QWORD *v30; // [rsp+10h] [rbp-90h]
  _QWORD *v31; // [rsp+10h] [rbp-90h]
  _QWORD *v32; // [rsp+10h] [rbp-90h]
  _QWORD *v33; // [rsp+10h] [rbp-90h]
  char v34; // [rsp+1Fh] [rbp-81h]
  char v35; // [rsp+1Fh] [rbp-81h]
  char v36; // [rsp+1Fh] [rbp-81h]
  char v37; // [rsp+1Fh] [rbp-81h]
  __int64 v38; // [rsp+20h] [rbp-80h]
  __int64 v39; // [rsp+20h] [rbp-80h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  __int64 v42; // [rsp+28h] [rbp-78h] BYREF
  __int64 v43; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v44; // [rsp+40h] [rbp-60h] BYREF
  __int64 v45; // [rsp+48h] [rbp-58h]
  _QWORD v46[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *((unsigned int *)a1 + 8);
  v42 = a2;
  if ( (_DWORD)v4 )
  {
    v5 = a1[2];
    v6 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( v2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return v7[1];
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v29 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( v2 == *v7 )
          goto LABEL_3;
        v10 = v29;
      }
    }
  }
  switch ( *(_WORD *)(v2 + 24) )
  {
    case 0:
    case 0xB:
      break;
    case 1:
      v28 = sub_1479A80(a1, *(_QWORD *)(v2 + 32));
      if ( v28 != *(_QWORD *)(v2 + 32) )
        v2 = sub_14835F0(*a1, v28, *(_QWORD *)(v2 + 40), 0);
      break;
    case 2:
      v20 = sub_1479A80(a1, *(_QWORD *)(v2 + 32));
      if ( v20 != *(_QWORD *)(v2 + 32) )
        v2 = sub_14747F0(*a1, v20, *(_QWORD *)(v2 + 40), 0);
      break;
    case 3:
      v24 = sub_1479A80(a1, *(_QWORD *)(v2 + 32));
      if ( v24 != *(_QWORD *)(v2 + 32) )
        v2 = sub_147B0D0(*a1, v24, *(_QWORD *)(v2 + 40), 0);
      break;
    case 4:
      v44 = v46;
      v45 = 0x200000000LL;
      v21 = *(_QWORD **)(v2 + 32);
      v32 = &v21[*(_QWORD *)(v2 + 40)];
      if ( v21 != v32 )
      {
        v36 = 0;
        v22 = *(_QWORD **)(v2 + 32);
        do
        {
          v40 = *v22;
          v43 = sub_1479A80(a1, *v22);
          sub_1458920((__int64)&v44, &v43);
          v13 = v44;
          ++v22;
          v36 |= v44[(unsigned int)v45 - 1] != v40;
        }
        while ( v32 != v22 );
        if ( v36 )
        {
          v23 = sub_147DD40(*a1, &v44, 0, 0);
          v13 = v44;
          v2 = v23;
        }
        goto LABEL_39;
      }
      break;
    case 5:
      v44 = v46;
      v45 = 0x200000000LL;
      v25 = *(_QWORD **)(v2 + 32);
      v33 = &v25[*(_QWORD *)(v2 + 40)];
      if ( v25 != v33 )
      {
        v37 = 0;
        v26 = *(_QWORD **)(v2 + 32);
        do
        {
          v41 = *v26;
          v43 = sub_1479A80(a1, *v26);
          sub_1458920((__int64)&v44, &v43);
          v13 = v44;
          ++v26;
          v37 |= v44[(unsigned int)v45 - 1] != v41;
        }
        while ( v33 != v26 );
        if ( v37 )
        {
          v27 = sub_147EE30(*a1, &v44, 0, 0);
          v13 = v44;
          v2 = v27;
        }
        goto LABEL_39;
      }
      break;
    case 6:
      v18 = sub_1479A80(a1, *(_QWORD *)(v2 + 32));
      v19 = sub_1479A80(a1, *(_QWORD *)(v2 + 40));
      if ( v18 != *(_QWORD *)(v2 + 32) || v19 != *(_QWORD *)(v2 + 40) )
        v2 = sub_1483CF0(*a1, v18, v19);
      break;
    case 7:
      if ( *(_QWORD *)(v2 + 48) == a1[5] )
        v2 = sub_1488A90(v2, *a1);
      else
        *((_BYTE *)a1 + 49) = 1;
      break;
    case 8:
      v44 = v46;
      v45 = 0x200000000LL;
      v15 = *(_QWORD **)(v2 + 32);
      v31 = &v15[*(_QWORD *)(v2 + 40)];
      if ( v15 != v31 )
      {
        v35 = 0;
        v16 = *(_QWORD **)(v2 + 32);
        do
        {
          v39 = *v16;
          v43 = sub_1479A80(a1, *v16);
          sub_1458920((__int64)&v44, &v43);
          v13 = v44;
          ++v16;
          v35 |= v44[(unsigned int)v45 - 1] != v39;
        }
        while ( v31 != v16 );
        if ( v35 )
        {
          v17 = sub_14813B0(*a1, &v44);
          v13 = v44;
          v2 = v17;
        }
        goto LABEL_39;
      }
      break;
    case 9:
      v44 = v46;
      v45 = 0x200000000LL;
      v11 = *(_QWORD **)(v2 + 32);
      v30 = &v11[*(_QWORD *)(v2 + 40)];
      if ( v11 != v30 )
      {
        v34 = 0;
        v12 = *(_QWORD **)(v2 + 32);
        do
        {
          v38 = *v12;
          v43 = sub_1479A80(a1, *v12);
          sub_1458920((__int64)&v44, &v43);
          v13 = v44;
          ++v12;
          v34 |= v44[(unsigned int)v45 - 1] != v38;
        }
        while ( v30 != v12 );
        if ( v34 )
        {
          v14 = sub_147A3C0(*a1, &v44);
          v13 = v44;
          v2 = v14;
        }
LABEL_39:
        if ( v13 != v46 )
          _libc_free((unsigned __int64)v13);
      }
      break;
    case 0xA:
      if ( !sub_146CEE0(*a1, v2, a1[5]) )
        *((_BYTE *)a1 + 48) = 1;
      break;
  }
  v43 = v2;
  sub_1466830((__int64)&v44, (__int64)(a1 + 1), &v42, &v43);
  return *(_QWORD *)(v46[0] + 8LL);
}
