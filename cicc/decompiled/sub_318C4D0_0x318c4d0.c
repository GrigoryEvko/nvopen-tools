// Function: sub_318C4D0
// Address: 0x318c4d0
//
_QWORD *__fastcall sub_318C4D0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r11
  unsigned __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // ebx
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v19; // rax
  _BYTE *v20; // r12
  __int64 v22; // rdx
  int v23; // r15d
  __int64 v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 **v29; // [rsp+0h] [rbp-70h]
  __int64 v30; // [rsp+0h] [rbp-70h]
  char v32[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v33; // [rsp+30h] [rbp-40h]

  v11 = (unsigned int)(a2 - 48);
  v12 = sub_318B710((__int64)a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v13 = *a1;
  v14 = *(_QWORD *)(a3 + 16);
  if ( (unsigned int)v11 > 0xC )
    BUG();
  if ( v13 == *(_QWORD *)(v14 + 8) )
  {
    v20 = *(_BYTE **)(a3 + 16);
    goto LABEL_9;
  }
  v15 = v12;
  v16 = *(_QWORD *)(v12 + 80);
  v17 = dword_44D1AC0[v11];
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v16 + 120LL);
  if ( v18 != sub_920130 )
  {
    v30 = v13;
    v28 = v18(v16, v17, (_BYTE *)v14, v13);
    v13 = v30;
    v20 = (_BYTE *)v28;
    goto LABEL_8;
  }
  if ( *(_BYTE *)v14 <= 0x15u )
  {
    v29 = (__int64 **)v13;
    if ( (unsigned __int8)sub_AC4810(v17) )
      v19 = sub_ADAB70(v17, v14, v29, 0);
    else
      v19 = sub_AA93C0(v17, v14, (__int64)v29);
    v13 = (__int64)v29;
    v20 = (_BYTE *)v19;
LABEL_8:
    if ( v20 )
      goto LABEL_9;
  }
  v33 = 257;
  v20 = (_BYTE *)sub_B51D30(v17, v14, v13, (__int64)v32, 0, 0);
  if ( (unsigned __int8)sub_920620((__int64)v20) )
  {
    v22 = *(_QWORD *)(v15 + 96);
    v23 = *(_DWORD *)(v15 + 104);
    if ( v22 )
      sub_B99FD0((__int64)v20, 3u, v22);
    sub_B45150((__int64)v20, v23);
  }
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v15 + 88) + 16LL))(
    *(_QWORD *)(v15 + 88),
    v20,
    a5,
    *(_QWORD *)(v15 + 56),
    *(_QWORD *)(v15 + 64));
  v24 = *(_QWORD *)v15;
  v25 = *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8);
  if ( v24 != v25 )
  {
    do
    {
      v26 = *(_QWORD *)(v24 + 8);
      v27 = *(_DWORD *)v24;
      v24 += 16;
      sub_B99FD0((__int64)v20, v27, v26);
    }
    while ( v25 != v24 );
    if ( (unsigned __int8)(*v20 - 67) <= 0xCu )
      return sub_3189EF0(a4, v20);
    return (_QWORD *)sub_31892C0(a4, (__int64)v20);
  }
LABEL_9:
  if ( (unsigned __int8)(*v20 - 67) <= 0xCu )
    return sub_3189EF0(a4, v20);
  return (_QWORD *)sub_31892C0(a4, (__int64)v20);
}
