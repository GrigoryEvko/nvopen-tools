// Function: sub_318C2A0
// Address: 0x318c2a0
//
_QWORD *__fastcall sub_318C2A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  int v11; // ebx
  __int64 v12; // rax
  unsigned __int8 *v13; // r15
  unsigned __int8 *v14; // r10
  __int64 v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // ebx
  __int64 (__fastcall *v18)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v19; // rax
  _BYTE *v20; // r12
  __int64 v22; // rdx
  int v23; // r15d
  __int64 v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rax
  unsigned __int8 *v29; // [rsp+0h] [rbp-70h]
  unsigned __int8 *v30; // [rsp+0h] [rbp-70h]
  char v32[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v33; // [rsp+30h] [rbp-40h]

  v11 = a1;
  v12 = sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v13 = *(unsigned __int8 **)(a3 + 16);
  v14 = *(unsigned __int8 **)(a2 + 16);
  v15 = v12;
  if ( (unsigned int)(a1 - 27) > 0x11 )
    BUG();
  v16 = *(_QWORD *)(v12 + 80);
  v17 = v11 - 14;
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v16 + 16LL);
  if ( v18 != sub_9202E0 )
  {
    v30 = *(unsigned __int8 **)(a2 + 16);
    v28 = v18(v16, v17, v14, v13);
    v14 = v30;
    v20 = (_BYTE *)v28;
    goto LABEL_8;
  }
  if ( *v14 <= 0x15u && *v13 <= 0x15u )
  {
    v29 = *(unsigned __int8 **)(a2 + 16);
    if ( (unsigned __int8)sub_AC47B0(v17) )
      v19 = sub_AD5570(v17, (__int64)v29, v13, 0, 0);
    else
      v19 = sub_AABE40(v17, v29, v13);
    v14 = v29;
    v20 = (_BYTE *)v19;
LABEL_8:
    if ( v20 )
      goto LABEL_9;
  }
  v33 = 257;
  v20 = (_BYTE *)sub_B504D0(v17, (__int64)v14, (__int64)v13, (__int64)v32, 0, 0);
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
    if ( (unsigned __int8)(*v20 - 42) <= 0x11u )
      return sub_3189D20(a4, v20);
    return (_QWORD *)sub_31892C0(a4, (__int64)v20);
  }
LABEL_9:
  if ( (unsigned __int8)(*v20 - 42) <= 0x11u )
    return sub_3189D20(a4, v20);
  return (_QWORD *)sub_31892C0(a4, (__int64)v20);
}
