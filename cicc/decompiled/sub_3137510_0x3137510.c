// Function: sub_3137510
// Address: 0x3137510
//
__int64 __fastcall sub_3137510(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rsi
  const void *v11; // r15
  size_t v12; // rbx
  bool v13; // di
  _QWORD *v14; // r8
  _BYTE *v15; // rax
  unsigned __int8 v16; // r8
  _QWORD *v17; // rax
  __int64 v18; // rax
  size_t v19; // rdx
  unsigned __int8 **v20; // rsi
  unsigned __int8 *v21; // rax
  unsigned __int8 v22; // dl
  __int64 v23; // rax
  __int64 v24; // rdi
  size_t v25; // rdx
  const void *v26; // rsi

  v6 = sub_B10CD0(a2);
  if ( v6 )
  {
    v7 = v6;
    v8 = *(_QWORD *)(a1 + 504);
    v9 = *(_BYTE *)(v7 - 16);
    v10 = v7 - 16;
    v11 = *(const void **)(v8 + 168);
    v12 = *(_QWORD *)(v8 + 176);
    v13 = (v9 & 2) != 0;
    if ( (v9 & 2) != 0 )
      v14 = *(_QWORD **)(v7 - 32);
    else
      v14 = (_QWORD *)(v10 - 8LL * ((v9 >> 2) & 0xF));
    v15 = (_BYTE *)*v14;
    if ( *(_BYTE *)*v14 == 16
      || ((v16 = *(v15 - 16), (v16 & 2) == 0)
        ? (v17 = &v15[-8 * ((v16 >> 2) & 0xF) - 16])
        : (v17 = (_QWORD *)*((_QWORD *)v15 - 4)),
          (v15 = (_BYTE *)*v17) != 0) )
    {
      if ( *((_QWORD *)v15 + 5) )
      {
        v18 = sub_B91420(*((_QWORD *)v15 + 5));
        v10 = v7 - 16;
        v12 = v19;
        v9 = *(_BYTE *)(v7 - 16);
        v11 = (const void *)v18;
        v13 = (v9 & 2) != 0;
      }
    }
    if ( v13 )
      v20 = *(unsigned __int8 ***)(v7 - 32);
    else
      v20 = (unsigned __int8 **)(v10 - 8LL * ((v9 >> 2) & 0xF));
    v21 = sub_AF34D0(*v20);
    v22 = *(v21 - 16);
    if ( (v22 & 2) != 0 )
      v23 = *((_QWORD *)v21 - 4);
    else
      v23 = (__int64)&v21[-8 * ((v22 >> 2) & 0xF) - 16];
    v24 = *(_QWORD *)(v23 + 16);
    if ( v24 )
    {
      v26 = (const void *)sub_B91420(v24);
      if ( v25 )
        return sub_3136EA0(a1, v26, v25, v11, v12, *(unsigned int *)(v7 + 4), *(unsigned __int16 *)(v7 + 2), a3);
    }
    else
    {
      v26 = 0;
    }
    v25 = 0;
    if ( a4 )
      v26 = sub_BD5D20(a4);
    return sub_3136EA0(a1, v26, v25, v11, v12, *(unsigned int *)(v7 + 4), *(unsigned __int16 *)(v7 + 2), a3);
  }
  return sub_3135D70(a1, a3);
}
