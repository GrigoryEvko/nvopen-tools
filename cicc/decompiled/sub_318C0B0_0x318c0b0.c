// Function: sub_318C0B0
// Address: 0x318c0b0
//
_QWORD *__fastcall sub_318C0B0(
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
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  _BYTE *v14; // r15
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 (__fastcall *v17)(__int64, unsigned int, _BYTE *); // rax
  _BYTE *v18; // r12
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rdx
  int v25; // ebx
  _BYTE v27[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v28; // [rsp+30h] [rbp-40h]

  v10 = sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v14 = *(_BYTE **)(a2 + 16);
  if ( (_DWORD)a1 != 26 )
    BUG();
  v15 = *(_QWORD *)(v10 + 80);
  v16 = v10;
  v17 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *))(*(_QWORD *)v15 + 48LL);
  if ( v17 != sub_9288C0 )
  {
    v18 = (_BYTE *)((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v17)(
                     v15,
                     12,
                     *(_QWORD *)(a2 + 16),
                     *(unsigned int *)(v16 + 104));
LABEL_5:
    if ( v18 )
      goto LABEL_6;
    goto LABEL_8;
  }
  if ( *v14 <= 0x15u )
  {
    v18 = (_BYTE *)sub_AAAFF0(12, *(unsigned __int8 **)(a2 + 16), v11, v12, v13);
    goto LABEL_5;
  }
LABEL_8:
  v28 = 257;
  v18 = (_BYTE *)sub_B50340(12, (__int64)v14, (__int64)v27, 0, 0);
  if ( (unsigned __int8)sub_920620((__int64)v18) )
  {
    v24 = *(_QWORD *)(v16 + 96);
    v25 = *(_DWORD *)(v16 + 104);
    if ( v24 )
      sub_B99FD0((__int64)v18, 3u, v24);
    sub_B45150((__int64)v18, v25);
  }
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v16 + 88) + 16LL))(
    *(_QWORD *)(v16 + 88),
    v18,
    a4,
    *(_QWORD *)(v16 + 56),
    *(_QWORD *)(v16 + 64));
  v20 = *(_QWORD *)v16;
  v21 = *(_QWORD *)v16 + 16LL * *(unsigned int *)(v16 + 8);
  if ( v20 != v21 )
  {
    do
    {
      v22 = *(_QWORD *)(v20 + 8);
      v23 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0((__int64)v18, v23, v22);
    }
    while ( v21 != v20 );
    if ( *v18 != 41 )
      return (_QWORD *)sub_31892C0(a3, (__int64)v18);
    return sub_3189C90(a3, v18);
  }
LABEL_6:
  if ( *v18 != 41 )
    return (_QWORD *)sub_31892C0(a3, (__int64)v18);
  return sub_3189C90(a3, v18);
}
