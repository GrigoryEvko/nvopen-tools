// Function: sub_1487400
// Address: 0x1487400
//
__int64 __fastcall sub_1487400(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // r9
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 *v15; // r15
  __int64 v16; // r10
  __int64 v17; // r14
  _QWORD *v18; // rcx
  __int64 v19; // r11
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // r14
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned int v26; // [rsp+1Ch] [rbp-94h]
  __int64 v27; // [rsp+20h] [rbp-90h]
  __int64 v28; // [rsp+20h] [rbp-90h]
  __int64 v29; // [rsp+28h] [rbp-88h]
  __int64 *v30; // [rsp+28h] [rbp-88h]
  __int64 *v31[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v32[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v33; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+58h] [rbp-58h]
  _QWORD v35[10]; // [rsp+60h] [rbp-50h] BYREF

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 **)(a2 - 8);
  else
    v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v29 = sub_146F1B0((__int64)a1, *v6);
  v7 = sub_1456040(v29);
  v8 = sub_1456E10((__int64)a1, v7);
  v26 = 4 * ((*(_BYTE *)(a2 + 17) & 2) != 0);
  v9 = sub_16348C0(a2);
  v10 = sub_1645D80(v9, 0);
  v11 = *(__int64 **)a3;
  v12 = v10;
  v33 = v35;
  v34 = 0x400000001LL;
  v13 = *(unsigned int *)(a3 + 8);
  v35[0] = v29;
  v30 = &v11[v13];
  if ( v11 == v30 )
  {
LABEL_13:
    v21 = (__int64)sub_147DD40((__int64)a1, (__int64 *)&v33, v26, 0, a4, a5);
    goto LABEL_20;
  }
  v14 = v12;
  v15 = v11;
  while ( 1 )
  {
    v16 = *v15;
    if ( *(_BYTE *)(v14 + 8) != 13 )
    {
      v14 = *(_QWORD *)(v14 + 24);
      v27 = *v15;
      v22 = sub_145D050((__int64)a1, v8, v14);
      v32[0] = sub_1483BD0(a1, v27, v8, a4, a5);
      v31[0] = v32;
      v32[1] = v22;
      v31[1] = (__int64 *)0x200000002LL;
      v23 = sub_147EE30(a1, v31, v26, 0, a4, a5);
      if ( v31[0] != v32 )
        _libc_free((unsigned __int64)v31[0]);
      v24 = (unsigned int)v34;
      if ( (unsigned int)v34 >= HIDWORD(v34) )
      {
        sub_16CD150(&v33, v35, 0, 8);
        v24 = (unsigned int)v34;
      }
      v33[v24] = v23;
      LODWORD(v34) = v34 + 1;
      goto LABEL_12;
    }
    if ( *(_WORD *)(v16 + 24) )
      break;
    v17 = *(_QWORD *)(v16 + 32);
    v18 = *(_QWORD **)(v17 + 24);
    if ( *(_DWORD *)(v17 + 32) > 0x40u )
      v18 = (_QWORD *)*v18;
    v19 = sub_145D250((__int64)a1, v8, v14, (unsigned int)v18);
    v20 = (unsigned int)v34;
    if ( (unsigned int)v34 >= HIDWORD(v34) )
    {
      v28 = v19;
      sub_16CD150(&v33, v35, 0, 8);
      v20 = (unsigned int)v34;
      v19 = v28;
    }
    v33[v20] = v19;
    LODWORD(v34) = v34 + 1;
    v14 = sub_1643D30(v14, v17);
LABEL_12:
    if ( v30 == ++v15 )
      goto LABEL_13;
  }
  v21 = sub_145DC80((__int64)a1, a2);
LABEL_20:
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  return v21;
}
